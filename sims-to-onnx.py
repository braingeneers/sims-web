"""
Exports a SIMS model to ONNX format as well as the gene list and class labels.
The model is based off of the core SIMS pytorch model exported to ONNX and then
extended with pre and post processing steps to match those in the SIMS source.
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.onnx
import anndata as ad
import onnx
from onnx import helper, TensorProto
import sclblonnx as so

from scsims import SIMS

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export a SIMS model to ONNX")
    parser.add_argument("checkpoint", type=str, help="Path to the checkpoint file")
    parser.add_argument("sample", type=str, help="Path to the sample for validation")
    args = parser.parse_args()

    model_name = args.checkpoint.split("/")[-1].split(".")[0]
    model_path = "/".join(args.checkpoint.split("/")[:-1])

    # Load the checkpoint
    print("Loading model...")
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
        f"{model_path}/{model_name}.core.onnx",
        training=torch.onnx.TrainingMode.EVAL,
        input_names=["input"],
        output_names=["logits","unknown"],
        export_params=True,
        opset_version=12,
    )
    print(f"Exported core model to {model_path}/{model_name}.core.onnx")

    # Get a normalized and inflated sample from the model for testing
    batch = next(enumerate(sims.model._parse_data(args.sample, batch_size=1)))
    sample = batch[1].to(torch.float32)

    # Write out the gene and class lists
    with open(f"{model_path}/{model_name}.genes", "w") as f:
        f.write("\n".join(map(str, sims.model.genes)))
    with open(f"{model_path}/{model_name}.classes", "w") as f:
        f.write("\n".join(map(str, sims.model.label_encoder.classes_)))
    print(f"Wrote out gene and classes lists to {model_path}/{model_name}")

    # Output a list of all models to populate the model selection drop down
    with open(f"{model_path}/models.txt", "w") as f:
        f.write("\n".join(list(set(f.split(".")[0] 
                                   for f in os.listdir("models") if f != "models.txt"))))

    """
    Build an onnx graph with post processing steps to match the SIMS model.
    See https://github.com/scailable/sclblonnx/tree/master/examples
    """

    # Preprocessing Graph
    pre_graph = so.empty_graph("preprocess")
    pre_graph = so.add_input(pre_graph, "input", "FLOAT", [1, model_input_size])
    n = so.node("LpNormalization", inputs=["input"], outputs=["lpnorm"])
    pre_graph = so.add_node(pre_graph, n)
    pre_graph = so.add_output(pre_graph, "lpnorm", "FLOAT", [1, model_input_size])

    # Validate the preprocessing graph against the SIMS model
    # result = so.run(g, inputs={"input": batch[1].to(torch.float32).numpy()}, outputs=["output"])
    adata = ad.read(args.sample)
    padded = torch.tensor(np.pad(adata.X[0, :], (0, model_input_size - adata.n_vars))).view((1, model_input_size))
    result = so.run(pre_graph, inputs={"input": padded.numpy()}, outputs=["lpnorm"])
    print("ONNX preprocess:", result[0][0][0:3])
    print("vs.")
    print("SIMS preprocess:", batch[1][0][[44, 54, 68]].numpy())

    # Load the core model
    core_graph = so.graph_from_file(f"{model_path}/{model_name}.core.onnx")
    core_graph = so.delete_output(core_graph, "unknown")

    # Merge pre with the core model
    g = so.merge(pre_graph, core_graph, io_match=[("lpnorm", "input")], _sclbl_check=False)
    so.list_inputs(g)
    so.list_outputs(g)
    so.check(g, _sclbl_check=True)

    # Try the new merged graph and save to disk
    result = so.run(g, inputs={"input": padded.numpy()}, outputs=["logits"])
    so.graph_to_file(g, f"{model_path}/{model_name}.onnx")

    # Load the new graph and add post processing steps
    g = so.graph_from_file(f"{model_path}/{model_name}.onnx")

    # Create a Constant node 'K' with value 3
    k_value = np.array([3], dtype=np.int64)
    k_node = helper.make_node(
        'Constant',
        inputs=[],
        outputs=['K'],
        value=helper.make_tensor(
            name='const_tensor_K',
            data_type=TensorProto.INT64,
            dims=[1],
            vals=k_value
        ),
        name='K_Node'
    )

    # Create the TopK node
    # Inputs: 'logits', 'K'
    # Outputs: 'topk_values', 'topk_indices'
    topk_node = helper.make_node(
        'TopK',
        inputs=['logits', 'K'],
        outputs=['topk_values', 'topk_indices'],
        name='TopK_Node',
        axis=-1,   # Optional: specify the axis over which to compute TopK
        largest=1, # Optional: set to 1 for largest values; 0 for smallest
        sorted=1   # Optional: set to 1 to return sorted values
    )

    # Create the Softmax node
    softmax_node = helper.make_node(
        'Softmax',
        inputs=['topk_values'],
        outputs=['probs'],
        name='Softmax_Node',
        axis=-1  # Apply softmax along the last dimension
    )

    # Add the new nodes to the graph
    g.node.extend([k_node, topk_node, softmax_node])

    # Optionally, add 'topk_values' and 'topk_indices' to the graph outputs
    g.output.extend([
        helper.make_tensor_value_info('topk_values', TensorProto.FLOAT, [None, 3]),
        helper.make_tensor_value_info('topk_indices', TensorProto.INT64, [None, 3]),
        helper.make_tensor_value_info('probs', TensorProto.FLOAT, [None, 3]),
    ])
    
    result = so.run(g, inputs={"input": padded.numpy()}, outputs=["topk_values"])
    so.graph_to_file(g, f"{model_path}/{model_name}.onnx")

    # """
    # Attempt to validate the onnx models with the full SIMS model
    # """
    # sample = batch[1].to(torch.float32)
    # res = sims.model(sample)[0][0]
    # probs, top_preds = res.topk(3)
    # probs = probs.softmax(dim=-1)
    # print("SIMS Python Model Results:")
    # print(probs)