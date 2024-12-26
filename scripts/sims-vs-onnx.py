"""
Compare SIMS vs. ONNX input and output
"""

import argparse
import torch
import anndata
import pandas as pd
import numpy as np

from scsims import SIMS

import torch.onnx

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare SIMS to ONNX")
    parser.add_argument("checkpoint", type=str, help="Path to a SIMS checkpoint")
    parser.add_argument("input", type=str, help="Path to h5ad to")
    parser.add_argument("predictions", type=str, help="CSV Predictions from sims-web")
    args = parser.parse_args()

    # Instantiate a model
    sims = SIMS(
        weights_path=args.checkpoint,
        map_location=torch.device("cpu"),
        weights_only=True,
    )

    # Load first 10 cells from a sample
    adata = anndata.read(args.input)
    adata = adata[0:10].copy()

    print("adata X format", )

    # Compare predictions of sims model and saved web csv
    sims_predictions = sims.predict(adata)
    web_predictions = pd.read_csv(args.csv)[0:10]

    assert (
        sims_predictions["prob_0"].round(2).equals(web_predictions["prob_0"].round(2))
    )
    assert sims_predictions["pred_0"].equals(web_predictions["pred_0"])

    print("Raw X before inflation")
    x = adata.X[0].toarray().flatten()
    print(np.nonzero(x)[0:10])
    print(x[np.nonzero(x)[0:10]])

    # Get inputs to model to compare manually to web code
    batch = next(enumerate(sims.model._parse_data(adata, batch_size=10)))[1]

    print("First inflated sample's non-zero indices and associated values:")
    x = batch.to(torch.float32)[0]
    print(np.nonzero(x).flatten()[0:10])
    print(x[np.nonzero(x).flatten()[0:10]])
