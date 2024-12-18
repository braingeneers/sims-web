"""
Exports a SIMS model to ONNX format as well as the gene list and class labels.
The model is based off of the core SIMS pytorch model exported to ONNX and then
extended with pre and post processing steps to match those in the SIMS source.
"""

import os
import argparse
import shutil
import numpy as np
import torch
import torch.onnx
import onnx
import sclblonnx as so

from scsims import SIMS

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export a SIMS model to ONNX")
    parser.add_argument("checkpoint", type=str, help="Path to the checkpoint file")
    parser.add_argument("destination", type=str, help="Path to save the ONNX model")
    args = parser.parse_args()

    model_id = args.checkpoint.split("/")[-1].split(".")[0]
    model_path = args.destination

    # Load the checkpoint
    print("Loading checkpoint...")
    sims = SIMS(
        weights_path=args.checkpoint,
        map_location=torch.device("cpu"),
        weights_only=True,
    )
    sims.model.eval()  # Turns off training mode?
    model_input_size = sims.model.input_dim
    num_model_genes = len(sims.model.genes)
    num_model_classes = len(sims.model.label_encoder.classes_)
    assert num_model_genes == model_input_size
    print(
        f"Loaded {args.checkpoint} with {num_model_genes} genes and {num_model_classes} classes"
    )

    # Export model to ONNX file so we can read back and add post-processing steps
    batch_size = 1
    torch.onnx.export(
        sims.model,
        torch.zeros(batch_size, num_model_genes),
        f"{model_path}/{model_id}.onnx",
        training=torch.onnx.TrainingMode.EVAL,
        input_names=["input"],
        output_names=["logits", "unknown"],
        export_params=True,
        opset_version=12,
    )
    print(f"Exported interium model to {model_path}/{model_id}.onnx")

    # Write out the gene and class lists
    with open(f"{model_path}/{model_id}.genes", "w") as f:
        f.write("\n".join(map(str, sims.model.genes)))
    with open(f"{model_path}/{model_id}.classes", "w") as f:
        f.write("\n".join(map(str, sims.model.label_encoder.classes_)))
    print(f"Wrote out gene and classes lists to {model_path}/{model_id}.genes/.classes")

    # Copy over the description file from the checkpoints directory
    shutil.copy(f"{os.path.dirname(args.checkpoint)}/{model_id}.json", args.destination)
    print(f"Copied over description file to {model_path}")

    # Output a list of all models to populate the model selection drop down
    with open(f"{model_path}/models.txt", "w") as f:
        f.write(
            "\n".join(
                list(
                    set(
                        f.split(".")[0]
                        for f in os.listdir("models")
                        if f != "models.txt" and f != ".DS_Store"
                    )
                )
            )
        )
    print(f"Wrote out list of all models to {model_path}/models.txt")

    """
    Build an onnx graph with post processing steps to match the SIMS model.
    See https://github.com/scailable/sclblonnx/tree/master/examples
    """

    # Load the core model back in via onnx
    core_model = onnx.load(f"{model_path}/{model_id}.onnx")
    core_graph = core_model.graph

    # core_graph = so.graph_from_file(f"{model_path}/{model_id}.onnx")
    # core_graph = so.delete_output(core_graph, "unknown")

    # Preprocessing Graph
    pre_graph = so.empty_graph("preprocess")
    pre_graph = so.add_input(pre_graph, "input", "FLOAT", [1, model_input_size])
    n = so.node("LpNormalization", inputs=["input"], outputs=["lpnorm"])
    pre_graph = so.add_node(pre_graph, n)
    pre_graph = so.add_output(pre_graph, "lpnorm", "FLOAT", [1, model_input_size])
    g = so.merge(
        pre_graph, core_graph, io_match=[("lpnorm", "input")], _sclbl_check=False
    )

    # Postprocessing Graph
    k_value = np.array([3], dtype=np.int64)
    k_node = onnx.helper.make_node(
        "Constant",
        inputs=[],
        outputs=["K"],
        value=onnx.helper.make_tensor(
            name="const_tensor_K",
            data_type=onnx.TensorProto.INT64,
            dims=[1],
            vals=k_value,
        ),
        name="K",
    )

    topk_node = onnx.helper.make_node(
        "TopK",
        inputs=["logits", "K"],
        outputs=["topk_values", "topk_indices"],
        name="TopK",
        axis=-1,  # Optional: specify the axis over which to compute TopK
        largest=1,  # Optional: set to 1 for largest values; 0 for smallest
        sorted=1,  # Optional: set to 1 to return sorted values
    )

    softmax_node = onnx.helper.make_node(
        "Softmax",
        inputs=["topk_values"],
        outputs=["probs"],
        name="Softmax",
        axis=-1,  # Apply softmax along the last dimension
    )

    # Add the new nodes to the graph
    g.node.extend([k_node, topk_node, softmax_node])

    # Add outputs
    g.output.extend(
        [
            onnx.helper.make_tensor_value_info(
                "topk_values", onnx.TensorProto.FLOAT, [None, 3]
            ),
            onnx.helper.make_tensor_value_info(
                "topk_indices", onnx.TensorProto.INT64, [None, 3]
            ),
            onnx.helper.make_tensor_value_info(
                "probs", onnx.TensorProto.FLOAT, [None, 3]
            ),
        ]
    )

    # Expose the encoding
    candidate = "/network/tabnet/ReduceSum_output_0"  # 32,
    shape_info = onnx.shape_inference.infer_shapes(core_model)
    for idx, node in enumerate(shape_info.graph.value_info):
        if node.name == candidate:
            print(idx, node)
            break
    assert node.name == candidate
    g.output.extend([node])
    g = so.rename_output(g, candidate, "encoding")

    # Define the candidate node names
    attentention_candidates = [
        "/network/tabnet/encoder/att_transformers.0/selector/Clip_output_0",
        "/network/tabnet/encoder/att_transformers.1/selector/Clip_output_0",
        "/network/tabnet/encoder/att_transformers.2/selector/Clip_output_0",
    ]

    # Gather the output names for the candidates
    shape_info = onnx.shape_inference.infer_shapes(core_model)
    attention_node_outputs = []
    for candidate in attentention_candidates:
        node_found = False
        for node in core_model.graph.node:
            for output_name in node.output:
                if output_name == candidate:
                    print(f"Found node output: {output_name}")
                    attention_node_outputs.append(output_name)
                    node_found = True
                    break
            if node_found:
                break
        else:
            raise Exception(f"Node output {candidate} not found in graph")

    if len(attention_node_outputs) != 3:
        raise Exception("Could not find all attention node outputs")

    # Create Add nodes to sum the outputs
    # First Add node to sum the first two outputs
    add_node1 = onnx.helper.make_node(
        "Add",
        inputs=[attention_node_outputs[0], attention_node_outputs[1]],
        outputs=["add_attention_1"],
        name="Add_Attention_1",
    )

    # Second Add node to add the third output
    add_node2 = onnx.helper.make_node(
        "Add",
        inputs=["add_attention_1", attention_node_outputs[2]],
        outputs=["attention"],
        name="Add_Attention_2",
    )

    # Add the new nodes to the graph
    g.node.extend([add_node1, add_node2])

    # Retrieve the data type and shape from one of the candidate outputs
    candidate_value_info = None
    for value_info in shape_info.graph.value_info:
        if value_info.name == attention_node_outputs[0]:
            candidate_value_info = value_info
            break

    if candidate_value_info is None:
        raise Exception("Could not find value info for the attention nodes")

    # Create the output tensor value info for 'attention'
    attention_output = onnx.helper.make_tensor_value_info(
        "attention",
        candidate_value_info.type.tensor_type.elem_type,
        [
            dim.dim_value if dim.HasField("dim_value") else None
            for dim in candidate_value_info.type.tensor_type.shape.dim
        ],
    )

    # Add 'attention' to the model outputs
    g.output.append(attention_output)

    print("Final outputs:")
    so.list_outputs(g)

    # Save the final graph
    print(
        f"Exporting graph with pre, core and post processing to {model_path}/{model_id}.onnx"
    )
    so.graph_to_file(g, f"{model_path}/{model_id}.onnx")
