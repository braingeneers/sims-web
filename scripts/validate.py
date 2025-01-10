"""
Compare SIMS vs. ONNX
"""

import argparse
import tempfile
import pandas as pd
import numpy as np
import torch
import torch.onnx
import anndata
import onnx
from onnxruntime import InferenceSession
from scsims import SIMS
import sclblonnx as so


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=str, help="Path to a SIMS checkpoint")
    parser.add_argument(
        "onnx", type=str, help="Path production ONNX model from checkpoint"
    )
    parser.add_argument("sample", type=str, help="Path to h5ad sample")
    parser.add_argument(
        "--count", type=int, help="Number of cells to compare (default = all)"
    )
    parser.add_argument(
        "--decimals", type=float, default=2, help="# decimals to compare"
    )
    args = parser.parse_args()

    # Load count cells from the sample
    adata = anndata.read(args.sample)[0 : args.count]

    print("Loading SIMS model...")
    sims = SIMS(
        weights_path=args.checkpoint,
        map_location=torch.device("cpu"),
        weights_only=True,
    )
    _ = sims.model.eval()  # Turns off training mode
    sm = sims.model

    print("Creating inflated normalized and non-normalized sample batch...")
    # The onnx model has lpnorm normalization built in so we need non-normalized
    # and normalized batches from the sample to compare to the raw onnx model
    batch_size = adata.shape[0]
    batch = next(enumerate(sm._parse_data(args.sample, batch_size, normalize=False)))
    x = batch[1].to(torch.float32)
    batch_normalized = next(
        enumerate(sm._parse_data(args.sample, batch_size, normalize=True))
    )
    x_normalized = batch_normalized[1].to(torch.float32)

    # Load production onnx model and extract opset version so we compare apples to
    # apples with the raw export
    pm = onnx.load(args.onnx)
    pm_opset_version = pm.opset_import[0].version if len(pm.opset_import) > 0 else None

    # Export a raw un-edited model to onnx to compare logits before
    # graph editing adds pre and post processing
    raw_model_path = tempfile.mktemp(suffix=".onnx")
    batch_size = 1

    torch.onnx.export(
        sims.model,
        torch.zeros(batch_size, len(sims.model.genes)),
        raw_model_path,
        training=torch.onnx.TrainingMode.EVAL,
        input_names=["input"],
        output_names=["logits", "unknown"],
        export_params=True,
        opset_version=pm_opset_version,
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
    )

    print("Loading core and production ONNX models...")
    cm = onnx.load(raw_model_path)

    opset_version = cm.opset_import[0].version if len(cm.opset_import) > 0 else None
    print("All ONNX models using opset version ", opset_version)

    # At this point we have sm, cm, and pm models loaded

    #
    # Compare raw logits
    #
    sm_logits = sm(x_normalized)[0].detach().numpy()

    cm_logits = so.run(
        cm.graph,
        inputs={"input": x_normalized.detach().numpy()},
        outputs=["logits"],
    )[0]

    pm_logits = so.run(
        pm.graph,
        inputs={"input": x.detach().numpy()},
        outputs=["logits"],
    )[0]

    print(
        "{} logits differ by more then {} decimals out of {}".format(
            np.count_nonzero(
                np.round(np.abs(sm_logits - pm_logits), decimals=args.decimals)
            ),
            args.decimals,
            pm_logits.shape[0] * pm_logits.shape[1],
        )
    )

    # assert np.allclose(sm_logits[0], cm_logits[0], rtol=0.001, atol=0.0)
    # assert np.allclose(sm_logits, cm_logits, rtol=0.1, atol=0.0, )

    # Load first 10 cells from a sample
    adata = anndata.read(args.sample)[0 : args.count]

    # Compare predictions of sims python and production onnx model
    p = sm.predict(adata)
    sm_probs = p.prob_0.values
    sm_preds = p.pred_0.values

    output = so.run(pm.graph, inputs={"input": x.detach().numpy()}, outputs=["probs"])
    pm_probs = [p[0] for p in output[0]]

    print(
        "{} prob_0 differ by more then {} decimals out of {}".format(
            np.count_nonzero(
                np.round(np.abs(sm_probs - pm_probs), decimals=args.decimals)
            ),
            args.decimals,
            len(pm_probs),
        )
    )

    # try:
    #     # Make sure cell id's and order match
    #     # Unfortunately the location of the cell ids in h5ad varies...
    #     # assert adata.obs.index.equals(web_predictions.index)
    #     # Make sure predictions and probabilites are close
    #     assert np.all(py_predictions.pred_0.values == web_predictions.pred_0.values)
    #     assert np.all(py_predictions.pred_1.values == web_predictions.pred_1.values)
    #     assert np.all(py_predictions.pred_2.values == web_predictions.pred_2.values)
    #     assert np.allclose(
    #         py_predictions.prob_0.values,
    #         web_predictions.prob_0.values,
    #         atol=args.tolerance,
    #     )
    #     assert np.allclose(
    #         py_predictions.prob_1.values,
    #         web_predictions.prob_1.values,
    #         atol=args.tolerance,
    #     )
    #     assert np.allclose(
    #         py_predictions.prob_2.values,
    #         web_predictions.prob_2.values,
    #         atol=args.tolerance,
    #     )
    # except AssertionError as e:
    #     print(e)
    #     print("Predictions do not match")
