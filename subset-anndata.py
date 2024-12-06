#!/usr/bin/env python3

import anndata
import numpy as np
import argparse
import sys

def sample_anndata(input_file: str, output_file: str, n_cells: int):
    """
    Reads an .h5ad AnnData file, randomly selects a subset of cells, and saves a new .h5ad file.

    Parameters:
        input_file (str): Path to the input .h5ad file.
        output_file (str): Path where the output .h5ad file will be saved.
        n_cells (int): Number of cells to randomly select.
    """
    # Read the input AnnData file
    adata = anndata.read_h5ad(input_file)
    
    # Total number of cells in the dataset
    total_cells = adata.n_obs
    
    # Check if requested number of cells is valid
    if n_cells > total_cells:
        raise ValueError(f"Requested number of cells ({n_cells}) exceeds total cells in the dataset ({total_cells}).")
    
    # Randomly select indices of cells to keep
    np.random.seed(42)  # Optional: Set seed for reproducibility
    selected_cells = np.random.choice(total_cells, size=n_cells, replace=False)
    
    # Subset the AnnData object to the selected cells
    adata_subset = adata[selected_cells, :].copy()
    
    # Save the subsetted AnnData object to a new file
    adata_subset.write_h5ad(output_file)

def main():
    parser = argparse.ArgumentParser(description='Sample a subset of cells from an AnnData .h5ad file.')
    parser.add_argument('input_file', type=str, help='Path to the input .h5ad file.')
    parser.add_argument('output_file', type=str, help='Path to the output .h5ad file.')
    parser.add_argument('--n_cells', '-n', type=int, default=1000, help='Number of cells to sample (default: 1000).')
    
    args = parser.parse_args()

    try:
        sample_anndata(args.input_file, args.output_file, args.n_cells)
        print(f"Successfully saved {args.n_cells} cells to {args.output_file}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()