#!/usr/bin/env python

import anndata as ad
import numpy as np
import argparse
import os
import sys


def subset_h5ad(input_path, output_path, n_cells=None, fraction=None):
    """
    Reads an AnnData object, selects a random subset of cells,
    and writes the subset to a new .h5ad file.

    Args:
        input_path (str): Path to the input .h5ad file.
        output_path (str): Path for the output .h5ad file.
        n_cells (int, optional): The exact number of cells to select.
                                 Defaults to None.
        fraction (float, optional): The fraction of total cells to select
                                    (e.g., 0.1 for 10%). Defaults to None.

    Raises:
        ValueError: If neither n_cells nor fraction is specified, or if both
                    are specified, or if the requested number/fraction is invalid.
        FileNotFoundError: If the input file does not exist.
    """
    # --- Input Validation ---
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if n_cells is None and fraction is None:
        raise ValueError("You must specify either --n_cells or --fraction.")
    if n_cells is not None and fraction is not None:
        raise ValueError("You cannot specify both --n_cells and --fraction.")
    if n_cells is not None and n_cells <= 0:
        raise ValueError("--n_cells must be a positive integer.")
    if fraction is not None and not (0 < fraction <= 1):
        raise ValueError("--fraction must be between 0 (exclusive) and 1 (inclusive).")

    # --- Read Data ---
    print(f"Reading AnnData object from: {input_path}")
    try:
        adata = ad.read_h5ad(input_path)
    except Exception as e:
        print(f"Error reading AnnData file: {e}", file=sys.stderr)
        sys.exit(1)

    total_cells = adata.n_obs
    print(f"Total cells found: {total_cells}")

    # --- Determine Number of Cells to Select ---
    if n_cells is not None:
        if n_cells > total_cells:
            print(
                f"Warning: Requested {n_cells} cells, but only {total_cells} available. "
                f"Selecting all {total_cells} cells.",
                file=sys.stderr,
            )
            num_to_select = total_cells
        else:
            num_to_select = n_cells
    elif fraction is not None:
        num_to_select = int(np.round(total_cells * fraction))
        if num_to_select == 0 and total_cells > 0:
            print(
                f"Warning: Fraction {fraction} resulted in 0 cells to select. "
                f"Selecting 1 cell instead.",
                file=sys.stderr,
            )
            num_to_select = 1  # Ensure at least one cell if possible
        elif (
            num_to_select > total_cells
        ):  # Should not happen with fraction <= 1, but safety check
            num_to_select = total_cells

    if num_to_select <= 0:
        print("Error: Number of cells to select is zero or less.", file=sys.stderr)
        sys.exit(1)

    print(f"Selecting {num_to_select} random cells...")

    # --- Select Random Subset ---
    # Ensure reproducibility if needed by setting np.random.seed() before this line
    # np.random.seed(42)
    random_indices = np.random.choice(
        adata.obs_names.values,  # Select from observation names (more robust than indices if adata is manipulated)
        size=num_to_select,
        replace=False,  # Ensure unique cells
    )

    # Subset the AnnData object
    # Using .obs_names ensures that if the AnnData object's internal
    # integer indexing changes, we still get the correct cells.
    adata_subset = adata[random_indices, :].copy()  # Use .copy() to avoid view issues

    print(f"Selected subset shape: {adata_subset.shape}")

    # --- Write Output ---
    print(f"Writing subset AnnData object to: {output_path}")
    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

        adata_subset.write_h5ad(output_path, compression="gzip")
    except Exception as e:
        print(f"Error writing output AnnData file: {e}", file=sys.stderr)
        sys.exit(1)

    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Select a random subset of cells from an AnnData .h5ad file."
    )
    parser.add_argument(
        "-i", "--input", required=True, help="Path to the input .h5ad file."
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Path for the output subset .h5ad file."
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-n", "--n_cells", type=int, help="The exact number of cells to select."
    )
    group.add_argument(
        "-f",
        "--fraction",
        type=float,
        help="The fraction of total cells to select (e.g., 0.1 for 10%).",
    )

    args = parser.parse_args()

    try:
        subset_h5ad(args.input, args.output, args.n_cells, args.fraction)
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

