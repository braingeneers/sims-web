import os
import re
import tempfile
import argparse
import numpy as np
import torch
import torch.onnx
import onnx
import onnxruntime
import anndata as ad
import sclblonnx as so
from scsims import SIMS

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explain an AnnData object using a SIMS model")
    parser.add_argument("checkpoint", type=str, help="Path to the checkpoint file")
    parser.add_argument("onnx", type=str, help="Path to the onnx file")
    parser.add_argument("sample", type=str, help="Path to the sample for validation")
    args = parser.parse_args()

    model_name = args.onnx.split("/")[-1].split(".")[0]
    dest = os.path.dirname(args.onnx)

    # Load the current production model
    model = onnx.load(args.onnx)
    g = model.graph

    # Expose the econding
    candidate = "/network/tabnet/Concat_output_0"  # 3 x 32
    # candidate = "/network/tabnet/ReduceSum_output_0"  # 32, 
    shape_info = onnx.shape_inference.infer_shapes(model)
    for idx, node in enumerate(shape_info.graph.value_info):
        if node.name == candidate:
            print(idx, node)
            break
    g.output.extend([node])
    so.list_outputs(g)
    g = so.rename_output(g, candidate, "encodings")
    so.list_outputs(g)

    # Expose the masks
    candidate = "/network/tabnet/encoder/att_transformers.2/selector/Clip_output_0"
    shape_info = onnx.shape_inference.infer_shapes(model)
    for idx, node in enumerate(shape_info.graph.value_info):
        if node.name == candidate:
            print(node)
            break
    model.graph.output.extend([node])
    g = so.rename_output(g, candidate, "mask")
    so.list_outputs(g)

    # Save the enhanced model
    so.graph_to_file(g, f"{dest}/{model_name}.explain.onnx")

    # Load back in for comparison
    model = onnx.load(f"{dest}/{model_name}.explain.onnx")
    onnx.checker.check_model(model)
    so.list_outputs(model.graph)

    # Load the checkpoint so we can compare to the onnx output
    sims = SIMS(weights_path=args.checkpoint, map_location=torch.device("cpu"), weights_only=True)
    sims.model.eval()  # Turns off training mode

    # Get an inflated sample
    batch = next(enumerate(sims.model._parse_data(args.sample, batch_size=1)))
    x = batch[1].to(torch.float32)

    onnx_logits, onnx_mask, onnx_encodings = so.run(model.graph, 
                    inputs={"input": x.detach().numpy()},
                    outputs=["logits", "mask", "encodings"])
    onnx_logits = onnx_logits[0]
    onnx_mask = onnx_mask[0]

    sims_logits = sims.model(x)[0][0].detach().numpy()

    # See if the logits are equivalent
    np.testing.assert_array_almost_equal(onnx_logits, sims_logits, decimal=3)


    # See if the encodings are equivalent
    sims_encodings = sims.model.network.tabnet.encoder(x)

    np.testing.assert_array_almost_equal(
        onnx_encodings[0][0],
        sims_encodings[0][0][0].detach().numpy(),
    )


    # # Get forward mask from tabnet
    # explain, masks = sims.model.network.forward_masks(x)
    # np.count_nonzero(explain.detach().numpy())


    # # Predict the cell types and get an explanation matrix
    # cell_predictions = sims.predict(args.sample)
    # explainability_matrix = sims.explain(args.sample)

    
    # a = explainability_matrix[0][0]
    # b = explain.detach().numpy()
    # np.max(np.setdiff1d(a, b))







    # embedded_x = sims.model.network.embedder(x)
    # a, b = sims.model.network.encoder(embedded_x)

    graph = so.graph_from_file("models/default.onnx")
