"""
Exports a SIMS model to ONNX format as well as the gene list and class labels.
The model is based off of the core SIMS pytorch model exported to ONNX and then
extended with pre and post processing steps to match those in the SIMS source.

NOTE: TabNet, used by SIMS, must be modified to run sparsemax in double precision
then exporting to ONNX to ensure concordance with SIMS python. See this commit:

https://github.com/rcurrie/tabnet/commit/812f13774ee8ea60b96d45a639c12f04185ab4a9
"""

import os
import argparse
import numpy as np
import torch
import torch.onnx
import onnx
from onnx import TensorProto
from onnx.helper import make_node, make_graph, make_tensor, make_tensor_value_info

import sclblonnx as so

from scsims import SIMS

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=str, help="Path to the checkpoint file")
    parser.add_argument("destination", type=str, help="Path to save the ONNX model")
    # NOTE: I've tried every other values of opset (13-17) and they all fail
    # to load correctly in the web runtime... so we're sticking with 12
    parser.add_argument(
        "--opset-version", type=int, default=12, help="ONNX opset version"
    )
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
    # Note variable batch size
    batch_size = 1
    torch.onnx.export(
        sims.model,
        torch.zeros(batch_size, num_model_genes),
        f"{model_path}/{model_id}.onnx",
        training=torch.onnx.TrainingMode.EVAL,
        input_names=["input"],
        output_names=["logits", "unknown"],
        export_params=True,
        opset_version=args.opset_version,
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
    )
    print(f"Exported interium model to {model_path}/{model_id}.onnx")

    # Write out the gene and class lists
    with open(f"{model_path}/{model_id}.genes", "w") as f:
        f.write("\n".join(map(str, sims.model.genes)))
    with open(f"{model_path}/{model_id}.classes", "w") as f:
        f.write("\n".join(map(str, sims.model.label_encoder.classes_)))
    print(f"Wrote out gene and classes lists to {model_path}/{model_id}.genes/.classes")

    """
    Build an onnx graph with post processing steps to match the SIMS model.
    See https://github.com/scailable/sclblonnx/tree/master/examples
    """

    # Load the core model back in via onnx
    model = onnx.load(f"{model_path}/{model_id}.onnx")
    opset_version = (
        model.opset_import[0].version if len(model.opset_import) > 0 else None
    )
    assert opset_version == args.opset_version
    print("opset_version:", opset_version)
    # Remove the "unknown" output if it exists
    for i, output in enumerate(model.graph.output):
        if output.name == "unknown":
            print(f"Removing output '{output.name}'")
            del model.graph.output[i]
            break
    assert so.graph_to_file(model.graph, f"{model_path}/{model_id}.onnx")
    model = onnx.load(f"{model_path}/{model_id}.onnx")

    # Preprocessing Graph
    pre_graph = so.empty_graph("preprocess")
    pre_graph = so.add_input(
        pre_graph, "input", "FLOAT", ["batch_size", model_input_size]
    )
    n = so.node("LpNormalization", inputs=["input"], outputs=["lpnorm"])
    pre_graph = so.add_node(pre_graph, n)
    pre_graph = so.add_output(
        pre_graph, "lpnorm", "FLOAT", ["batch_size", model_input_size]
    )
    g = so.merge(
        pre_graph, model.graph, io_match=[("lpnorm", "input")], _sclbl_check=False
    )

    # Note we can't assign g to model.graph so we save and reload
    assert so.graph_to_file(g, f"{model_path}/{model_id}.onnx")
    model = onnx.load(f"{model_path}/{model_id}.onnx")

    # Create a post processing subgraph with a dynamic batch dimension
    k_value = np.array([3], dtype=np.int64)
    k_node = make_node(
        "Constant",
        inputs=[],
        outputs=["K"],
        value=make_tensor(
            name="const_tensor_K",
            data_type=TensorProto.INT64,
            dims=[1],
            vals=k_value,
        ),
        name="K",
    )
    topk_node = make_node(
        "TopK",
        inputs=["logits", "K"],
        outputs=["topk_values", "topk_indices"],
        name="TopK",
        axis=-1,  # Optional: specify the axis over which to compute TopK
        largest=1,  # Optional: set to 1 for largest values; 0 for smallest
        sorted=1,  # Optional: set to 1 to return sorted values
    )
    softmax_node = make_node(
        "Softmax",
        inputs=["topk_values"],
        outputs=["probs"],
        name="Softmax",
        axis=-1,  # Apply softmax along the last dimension
    )
    # Add the new nodes to the graph
    model.graph.node.extend([k_node, topk_node, softmax_node])
    # Add outputs
    model.graph.output.extend(
        [
            make_tensor_value_info(
                "topk_values", onnx.TensorProto.FLOAT, ["batch_size", 3]
            ),
            make_tensor_value_info(
                "topk_indices", onnx.TensorProto.INT64, ["batch_size", 3]
            ),
            make_tensor_value_info("probs", onnx.TensorProto.FLOAT, ["batch_size", 3]),
        ]
    )

    assert so.graph_to_file(model.graph, f"{model_path}/{model_id}.onnx")

    # Expose the encoding
    candidate = "/network/tabnet/ReduceSum_output_0"  # 32,
    shape_info = onnx.shape_inference.infer_shapes(model)
    for idx, node in enumerate(shape_info.graph.value_info):
        if node.name == candidate:
            print(idx, node)
            break
    assert node.name == candidate
    model.graph.output.extend([node])

    _ = so.rename_output(model.graph, candidate, "encoding")

    # Define the candidate mask/explainability outputs
    attentention_candidates = [
        "/network/tabnet/encoder/att_transformers.0/selector/Clip_output_0",
        "/network/tabnet/encoder/att_transformers.1/selector/Clip_output_0",
        "/network/tabnet/encoder/att_transformers.2/selector/Clip_output_0",
    ]
    # Gather the output names for the candidates
    shape_info = onnx.shape_inference.infer_shapes(model)
    attention_node_outputs = []
    for candidate in attentention_candidates:
        node_found = False
        for node in model.graph.node:
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
    model.graph.node.extend([add_node1, add_node2])

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
    model.graph.output.append(attention_output)

    print("Final iputs:")
    so.list_inputs(model.graph)
    print("Final outputs:")
    so.list_outputs(model.graph)

    # Save the final graph
    print(
        f"Exporting graph with pre, core and post processing using ONNX opset version {args.opset_version} to {model_path}/{model_id}.onnx"
    )
    # onnx.save(model, f"{model_path}/{model_id}.onnx")
    so.graph_to_file(
        model.graph,
        f"{model_path}/{model_id}.onnx",
        onnx_opset_version=args.opset_version,
    )

    # Count the total number of parameters in the model
    total_params = 0
    for init in model.graph.initializer:
        # init.dims is a list of tensor dimensions
        tensor_size = np.prod(init.dims)
        total_params += tensor_size
    print("Total number of parameters in the onnx model:", total_params)
