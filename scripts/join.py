import anndata as ad
import pandas as pd


def join_cell_labels(
    h5ad_path: str,
    csv_path: str,
    cell_id_col: str = "cell_id",
    label_col: str = "label",
    new_obs_col: str = "cell_label",
) -> ad.AnnData:
    """
    Load an AnnData object from an h5ad file and join cell labels from a CSV file into adata.obs.

    The CSV file must contain a column with cell IDs (default 'cell_id') that match adata.obs.index
    and a column with cell labels (default 'label'). The labels are added to adata.obs using the key
    specified by 'new_obs_col'.

    Parameters:
        h5ad_path (str): Path to the .h5ad file.
        csv_path (str): Path to the CSV file containing cell labels.
        cell_id_col (str): Name of the CSV column with cell IDs. Default is "cell_id".
        label_col (str): Name of the CSV column with cell labels. Default is "label".
        new_obs_col (str): New column name for the cell labels in adata.obs. Default is "cell_label".

    Returns:
        anndata.AnnData: The updated AnnData object with the cell labels in adata.obs.
    """
    # Load the AnnData object from the h5ad file.
    adata = ad.read_h5ad(h5ad_path)

    # Load the CSV as a DataFrame.
    df_labels = pd.read_csv(csv_path)

    # Ensure the required columns exist.
    if cell_id_col not in df_labels.columns:
        raise ValueError(f"CSV file does not contain the cell id column: {cell_id_col}")
    if label_col not in df_labels.columns:
        raise ValueError(f"CSV file does not contain the label column: {label_col}")

    # Set the cell_id column as index for easier alignment.
    df_labels = df_labels.set_index(cell_id_col)

    # Reindex to ensure that cells match the AnnData obs.index, then assign to obs.
    adata.obs[new_obs_col] = df_labels[label_col].reindex(adata.obs.index)

    return adata


# Example usage:
if __name__ == "__main__":
    h5ad_file = "data/human.h5ad"
    csv_file = "data/human_labels.csv"

    updated_adata = join_cell_labels(
        h5ad_file,
        csv_file,
        cell_id_col="sample_name",
        label_col="class_label",
        new_obs_col="cell_label",
    )

    # Save the updated AnnData object if needed.
    updated_adata.write("data/human_with_labels.h5ad")
    print("Cell labels joined successfully.")
