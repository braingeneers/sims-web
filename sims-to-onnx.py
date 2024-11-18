import argparse
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
    parser = argparse.ArgumentParser(description="Export a SIMS model to ONNX")
    parser.add_argument("checkpoint", type=str, help="Path to the checkpoint file")
    args = parser.parse_args()

    # Load the checkpoint
    print("Loading model...")
    sims = SIMS(weights_path=args.checkpoint, map_location=torch.device("cpu"))
    model_name = args.checkpoint.split("/")[-1].split(".")[0]
    model_path = "/".join(args.checkpoint.split("/")[:-1])
    num_genes = len(sims.model.genes)
    print(f"Loaded model {model_name} with {num_genes} genes")

    # Export model to ONNX file
    # wrt batch size https://github.com/microsoft/onnxruntime/issues/19452#issuecomment-1947799229
    batch_size = 1
    sims.model.to_onnx(
        f"{model_path}/{model_name}.onnx",
        torch.zeros(batch_size, num_genes),
        export_params=True,
    )
    print(f"Exported model to {model_path}/{model_name}.onnx")

    # Write out the list of genes corresponding to the models input
    with open(f"{model_path}/{model_name}.genes.txt", "w") as f:
        f.write("\n".join(map(str, sims.model.genes)))
    print(f"Wrote out gene list to {model_path}/{model_name}.genes.txt")
