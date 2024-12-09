"""
Exports a SIMS model to ONNX format as well as the gene list and class labels.
The model is based off of the core SIMS pytorch model exported to ONNX and then
extended with pre and post processing steps to match those in the SIMS source.
"""

import os
import argparse
import numpy as np
import torch
import torch.onnx
from onnx import helper, TensorProto
import sclblonnx as so

from scsims import SIMS

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export a SIMS model to ONNX")
    parser.add_argument("checkpoint", type=str, help="Path to the checkpoint file")
    parser.add_argument("destination", type=str, help="Path to save the ONNX model")

    args = parser.parse_args()

    model_name = args.checkpoint.split("/")[-1].split(".")[0]
    model_path = args.destination

    # Load the checkpoint
    print("Loading checkpoint...")
    sims = SIMS(weights_path=args.checkpoint, map_location=torch.device("cpu"), weights_only=True)
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
        f"{model_path}/{model_name}.onnx",
        training=torch.onnx.TrainingMode.EVAL,
        input_names=["input"],
        output_names=["logits","unknown"],
        export_params=True,
        opset_version=12,
    )
    print(f"Exported interium model to {model_path}/{model_name}.onnx")

    # Write out the gene and class lists
    with open(f"{model_path}/{model_name}.genes", "w") as f:
        f.write("\n".join(map(str, sims.model.genes)))
    with open(f"{model_path}/{model_name}.classes", "w") as f:
        f.write("\n".join(map(str, sims.model.label_encoder.classes_)))
    print(f"Wrote out gene and classes lists to {model_path}/{model_name}.genes/.classes")

    # Output a list of all models to populate the model selection drop down
    with open(f"{model_path}/models.txt", "w") as f:
        f.write("\n".join(list(set(f.split(".")[0] 
                                   for f in os.listdir("models") if f != "models.txt"))))
    print(f"Wrote out list of all models to {model_path}/models.txt")

    """
    Build an onnx graph with post processing steps to match the SIMS model.
    See https://github.com/scailable/sclblonnx/tree/master/examples
    """

    # Load the core model back in via onnx
    core_graph = so.graph_from_file(f"{model_path}/{model_name}.onnx")
    core_graph = so.delete_output(core_graph, "unknown")

    # Preprocessing Graph
    pre_graph = so.empty_graph("preprocess")
    pre_graph = so.add_input(pre_graph, "input", "FLOAT", [1, model_input_size])
    n = so.node("LpNormalization", inputs=["input"], outputs=["lpnorm"])
    pre_graph = so.add_node(pre_graph, n)
    pre_graph = so.add_output(pre_graph, "lpnorm", "FLOAT", [1, model_input_size])
    g = so.merge(pre_graph, core_graph, io_match=[("lpnorm", "input")], _sclbl_check=False)

    # Postprocessing Graph
    k_value = np.array([3], dtype=np.int64)
    k_node = helper.make_node(
        'Constant',
        inputs=[], outputs=['K'],
        value=helper.make_tensor(
            name='const_tensor_K',
            data_type=TensorProto.INT64,
            dims=[1],
            vals=k_value
        ),
        name='K'
    )

    topk_node = helper.make_node(
        'TopK',
        inputs=['logits', 'K'], outputs=['topk_values', 'topk_indices'],
        name='TopK',
        axis=-1,   # Optional: specify the axis over which to compute TopK
        largest=1, # Optional: set to 1 for largest values; 0 for smallest
        sorted=1   # Optional: set to 1 to return sorted values
    )

    softmax_node = helper.make_node(
        'Softmax',
        inputs=['topk_values'], outputs=['probs'],
        name='Softmax',
        axis=-1  # Apply softmax along the last dimension
    )

    # Add the new nodes to the graph
    g.node.extend([k_node, topk_node, softmax_node])

    # Add outputs
    g.output.extend([
        helper.make_tensor_value_info('topk_values', TensorProto.FLOAT, [None, 3]),
        helper.make_tensor_value_info('topk_indices', TensorProto.INT64, [None, 3]),
        helper.make_tensor_value_info('probs', TensorProto.FLOAT, [None, 3]),
    ])
    
    # Save the final graph
    print(f"Exporting graph with pre, core and post processing to {model_path}/{model_name}.onnx")
    so.graph_to_file(g, f"{model_path}/{model_name}.onnx")