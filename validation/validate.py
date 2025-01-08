"""
Compare SIMS vs. ONNX
"""

import argparse
import torch
import anndata
import pandas as pd
import numpy as np

import onnx
from onnxruntime import InferenceSession

from scsims import SIMS

import torch.onnx

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=str, help="Path to a SIMS checkpoint")
    parser.add_argument("sample", type=str, help="Path to h5ad")
    parser.add_argument("--count", type=int, help="Number of cells to compare")
    parser.add_argument(
        "--tolerance", type=float, default=1e-3, help="Tolerance for comparison"
    )
    parser.add_argument("predictions", type=str, help="sims-web csv predictions")
    args = parser.parse_args()

    print("Instantiating SIMS model...")
    sims = SIMS(
        weights_path=args.checkpoint,
        map_location=torch.device("cpu"),
        weights_only=True,
    )
    _ = sims.model.eval()  # Turns off training mode

    #
    # Compare the raw logits of py sims to py onnx
    #
    batch = next(enumerate(sims.model._parse_data(args.sample, batch_size=10)))
    x = batch[1].to(torch.float32)
    np.count_nonzero(x.detach().numpy())

    sims_logigs = sims.model(x)


    model = onnx.load(args.onnx)

    # Load first 10 cells from a sample
    adata = anndata.read(args.sample)[0 : args.count]

    # Compare predictions of sims model and saved web csv
    py_predictions = sims.predict(adata)
    web_predictions = pd.read_csv(args.predictions, index_col="cell_id")[0 : args.count]

    try:
        # Make sure cell id's and order match
        # Unfortunately the location of the cell ids in h5ad varies...
        # assert adata.obs.index.equals(web_predictions.index)
        # Make sure predictions and probabilites are close
        assert np.all(py_predictions.pred_0.values == web_predictions.pred_0.values)
        assert np.all(py_predictions.pred_1.values == web_predictions.pred_1.values)
        assert np.all(py_predictions.pred_2.values == web_predictions.pred_2.values)
        assert np.allclose(
            py_predictions.prob_0.values,
            web_predictions.prob_0.values,
            atol=args.tolerance,
        )
        assert np.allclose(
            py_predictions.prob_1.values,
            web_predictions.prob_1.values,
            atol=args.tolerance,
        )
        assert np.allclose(
            py_predictions.prob_2.values,
            web_predictions.prob_2.values,
            atol=args.tolerance,
        )
    except AssertionError as e:
        print(e)
        print("Predictions do not match")
