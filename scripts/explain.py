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
    parser = argparse.ArgumentParser(
        description="Explain an AnnData object using a SIMS model"
    )
    parser.add_argument("checkpoint", type=str, help="Path to the checkpoint file")
    parser.add_argument("onnx", type=str, help="Path to the onnx file")
    parser.add_argument("sample", type=str, help="Path to the sample for validation")
    args = parser.parse_args()

    model_name = args.onnx.split("/")[-1].split(".")[0]
    dest = os.path.dirname(args.onnx)

    # Load the checkpoint so we can compare to the onnx output
    sims = SIMS(
        weights_path=args.checkpoint,
        map_location=torch.device("cpu"),
        weights_only=True,
    )
    _ = sims.model.eval()  # Turns off training mode

    # Get an inflated sample
    batch = next(enumerate(sims.model._parse_data(args.sample, batch_size=1)))
    x = batch[1].to(torch.float32)

    # Get forward mask from tabnet
    M_explain, sims_masks = sims.model.network.forward_masks(x)

    # Expose the attention mask 0
    onnx_masks = []
    for i, candidate in enumerate([
        "/network/tabnet/encoder/att_transformers.0/selector/Clip_output_0",
        "/network/tabnet/encoder/att_transformers.1/selector/Clip_output_0",
        "/network/tabnet/encoder/att_transformers.2/selector/Clip_output_0",
    ]):
        # Load the current production model
        print(candidate)
        model = onnx.load(args.onnx)
        g = model.graph
        shape_info = onnx.shape_inference.infer_shapes(model)
        for idx, node in enumerate(shape_info.graph.value_info):
            if node.name == candidate:
                # print(node)
                break
        assert node.name == candidate
        model.graph.output.extend([node])
        g = so.rename_output(g, candidate, "mask")
        onnx_mask = so.run(
            model.graph,
            inputs={"input": x.detach().numpy()},
            outputs=["mask"],
        )
        onnx_masks.append(onnx_mask[0])

        # Compare the first mask
        np.count_nonzero(sims_masks[0][0].detach().numpy())
        np.count_nonzero(onnx_masks[0])
        np.nonzero(sims_masks[0][0].detach().numpy())
        np.nonzero(onnx_masks[0])
        np.testing.assert_array_almost_equal(
            sims_masks[0][0].detach().numpy(), onnx_masks[0][0], decimal=5
        )

        i = 2
        np.nonzero(sims_masks[i][0].detach().numpy())
        np.nonzero(onnx_masks[i][0])

    # # Save the enhanced model
    # so.graph_to_file(g, f"{dest}/{model_name}.explain.onnx")

    # # Load back in for comparison
    # model = onnx.load(f"{dest}/{model_name}.explain.onnx")
    # onnx.checker.check_model(model)
    # so.list_outputs(model.graph)

    # sims_logits = sims.model(x)[0][0].detach().numpy()

    # # See if the logits are equivalent
    # np.testing.assert_array_almost_equal(onnx_logits, sims_logits, decimal=3)

    # # See if the encoding are equivalent
    # sims_encoding = sims.model.network.tabnet.encoder(x)
    # np.testing.assert_array_almost_equal(
    #     onnx_encoding[0][0],
    #     sims_encoding[0][0][0].detach().numpy(),
    # )

    # # sims_top_ind = (-sims_masks[0].detach().numpy()).argsort()[:4]

    # # np.count_nonzero(explain.detach().numpy())

    # # # Predict the cell types and get an explanation matrix
    # # cell_predictions = sims.predict(args.sample)
    # # explainability_matrix = sims.explain(args.sample)

    # # a = explainability_matrix[0][0]
    # # b = explain.detach().numpy()
    # # np.max(np.setdiff1d(a, b))

    # # embedded_x = sims.model.network.embedder(x)
    # # a, b = sims.model.network.encoder(embedded_x)

    # # graph = so.graph_from_file("models/default.onnx")
