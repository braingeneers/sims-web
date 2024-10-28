import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import torch
import gc

import sys

# Assumes you have the SIMS repo as a peer to this one...
sys.path.insert(0, "../SIMS")
from scsims import SIMS

import torch.onnx

if __name__ == "__main__":
    # Load the checkpoint
    sims = SIMS(
        weights_path="data/11A_2organoids.ckpt", map_location=torch.device("cpu")
    )

    # Save the model as an ONNX - note constant batch size at this point
    batch_size = 8
    sims.model.to_onnx("data/sims.onnx", torch.zeros(batch_size, 33694), export_params=True)
    print("Wrote out ONNX model")

    # Write out the list of genes corresponding to the models input
    with open("data/genes.txt", "w") as f:
        f.write("\n".join(map(str, sims.model.genes)))
    print("Wrote out gene list")
