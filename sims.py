import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import torch
import gc

import sys

sys.path.insert(0, "../SIMS")
from scsims import SIMS


if __name__ == "__main__":
    # Load the dataset
    adata = sc.read_h5ad("data/pbmc3k_processed.h5ad")
    # Process the dataset
    sc.pp.scale(adata)

    # Load the checkpoint
    sims = SIMS(
        weights_path="data/11A_2organoids.ckpt", map_location=torch.device("cpu")
    )

    # Perform inference
    cell_predictions = sims.predict(adata, num_workers=0)

    # Join cell predictions with adata
    adata.obs.reset_index(
        inplace=True
    )  # This is so that the adata.obs index is the same as the cell predictions index
    adata.obs = adata.obs.join(cell_predictions)
    # Save the predictions
    # adata.obs.to_csv(f"predictions/{adata_name}-{checkpoint_name}.csv", index=False)
