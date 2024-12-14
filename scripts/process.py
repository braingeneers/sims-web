"""
Process a raw h5 file and output an h5ad file
"""

import argparse
import scanpy as sc
import numpy as np
import pandas as pd
import torch
from scsims import SIMS

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process a raw h5 file outputing an h5ad file"
    )
    parser.add_argument(
        "input_file", type=str, help="Path to the input .h5 or .h5ad file."
    )
    parser.add_argument("output_file", type=str, help="Path to the output .h5ad file.")
    parser.add_argument("--checkpoint", type=str, help="Path to the checkpoint file")
    parser.add_argument("--num_cells", "-n", type=int, help="Number of cells to sample")
    parser.add_argument(
        "--pre_process", "-p", action="store_true", help="Pre-process the data"
    )
    parser.add_argument(
        "--label", "-l", action="store_true", help="Label the cells with SIMS"
    )

    args = parser.parse_args()

    adata = None
    if args.input_file.endswith(".h5"):
        adata = sc.read_10x_h5(args.input_file)
    else:
        adata = sc.read(args.input_file)

    # Select a random subset of the data if subset_size is specified
    if args.num_cells is not None:
        if args.num_cells > adata.n_obs:
            raise ValueError(
                f"Requested subset size ({args.subset_size}) is greater than the total number of cells ({adata.n_obs})."
            )
        np.random.seed(42)  # For reproducibility
        subset_indices = np.random.choice(
            adata.n_obs, size=args.num_cells, replace=False
        )
        adata = adata[subset_indices, :].copy()

    if args.pre_process:
        print("Pre-processing data...")
        # Load and pre-process raw data
        adata = sc.read_10x_h5(args.input_file)
        sc.pp.filter_cells(adata, min_genes=100)
        sc.pp.filter_genes(adata, min_cells=3)
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        sc.pp.scale(adata)

    if args.label:
        print("Labeling cells with SIMS...")
        sims = SIMS(
            weights_path=args.checkpoint,
            map_location=torch.device("cpu"),
            weights_only=True,
        )
        predictions = sims.predict(adata)
        adata.obs['cell_type'] = predictions['pred_0'].astype('category').values


    # Write the processed data to an h5ad file
    print(f"Writing processed data to {args.output_file}...")
    sc.write(args.output_file, adata)
