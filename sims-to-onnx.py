import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import torch
import gc

import sys

sys.path.insert(0, "../SIMS")
from scsims import SIMS

import torch.onnx

if __name__ == "__main__":
    # Load a sample dataset - onnx requires an example
    adata = sc.read_h5ad("data/pbmc3k_processed.h5ad")
    # Process the dataset
    sc.pp.scale(adata)

    # Load the checkpoint
    sims = SIMS(
        weights_path="data/11A_2organoids.ckpt", map_location=torch.device("cpu")
    )

    # Save the model as an ONNX
    sims.model.to_onnx("data/sims.onnx", torch.tensor(adata.X), export_params=True)

    sims.model.to_onnx("data/sims.onnx", torch.zeros(2, 33694), export_params=True)
