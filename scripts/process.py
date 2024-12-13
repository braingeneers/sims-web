"""
Process a raw h5 file and output an h5ad file
"""
import argparse
import scanpy as sc
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a raw h5 file outputing an h5ad file")
    parser.add_argument("sample", type=str, help="Path to the sample file")
    parser.add_argument("--subset-size", type=int, default=None, help="Number of cells to include in the subset")
    args = parser.parse_args()

    # Read the raw h5 file
    adata = sc.read_10x_h5(args.sample)

    # Preprocess the data
    sc.pp.filter_cells(adata, min_genes=100)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.scale(adata)

    # Select a random subset of the data if subset_size is specified
    if args.subset_size is not None:
        if args.subset_size > adata.n_obs:
            raise ValueError(f"Requested subset size ({args.subset_size}) is greater than the total number of cells ({adata.n_obs}).")
        np.random.seed(42)  # For reproducibility
        subset_indices = np.random.choice(adata.n_obs, size=args.subset_size, replace=False)
        adata = adata[subset_indices, :].copy()

    # Write the processed data to an h5ad file
    sc.write(args.sample + "ad", adata)