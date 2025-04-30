#!/usr/bin/env python
"""
Single cell expression data encoder and clustering script.

This script provides functionality to encode and cluster single cell expression data
using ONNX models with PyTorch computation backend.
"""

from pathlib import Path
from typing import Optional, Dict

import numpy as np
import torch
import typer
import onnxruntime as ort
from umap_pytorch import PUMAP
import anndata as ad
import scanpy as sc
from tqdm import tqdm

# Create Typer app
app = typer.Typer(help="Single cell expression data encoder and clustering")


@app.command()
def predict(
    onnx_model_path: Path = typer.Argument(..., help="Path to model .onnx file"),
    sample_path: Path = typer.Argument(..., help="Path to sample.h5ad file"),
    cell_type_field: Optional[str] = typer.Option(
        "subclass_label",
        help="AnnData obs field containing cell type labels (default: subclass_label)",
    ),
    batch_size: int = typer.Option(
        32, help="Number of samples to process in each batch"
    ),
    num_samples: Optional[int] = typer.Option(
        None, help="Limit the total number of inputs processed, process all if None"
    ),
) -> None:
    """
    Generate predictions and encodings for a sample .h5ad file containing single cell expression data.

    Args:
        onnx_model_path: Path to an model .onnx file
        sample_path: Path to a sample.h5ad file containing gene expression data
        cell_type_field: AnnData obs field containing cell type labels (default: subclass_label)
        batch_size: Number of samples to process at a time
        num_samples: Limit the total number of inputs processed, process all if None
    """
    typer.echo(
        f"Generating predictions and encodings for {sample_path} using model {onnx_model_path}"
    )

    # Instantiate the model
    model_session = ort.InferenceSession(str(onnx_model_path))

    # Get model input shape to determine the expected genes
    input_shape = model_session.get_inputs()[0].shape
    model_input_size = input_shape[1] if len(input_shape) > 1 else input_shape[0]

    # Load the sample data and preprocess it
    typer.echo(f"Loading sample data and preprocessing from {sample_path}...")
    adata = ad.read_h5ad(sample_path)
    sc.pp.filter_cells(adata, min_genes=100)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata)
    ### Logarithmizing the data
    sc.pp.log1p(adata)
    sc.pp.scale(adata)

    # Create an inflation mapping between model genes and sample genes
    inflation_map = create_inflation_map(adata, onnx_model_path)

    # Read the classes file for the model
    classes_path = Path(onnx_model_path).with_suffix(".classes")
    with open(classes_path, "r") as f:
        classes = [line.strip() for line in f]

    typer.echo(f"Loaded {len(classes)} classes from {classes_path}")

    # Create a mapping from class name to index
    class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}

    # Determine number of cells to process
    num_cells = num_samples if num_samples is not None else adata.n_obs
    if num_samples is not None:
        num_cells = min(adata.n_obs, num_samples)

    typer.echo(f"Processing {num_cells} cells with batch size {batch_size}")

    # Initialize numpy array to store encodings
    predictions = []
    encodings = []
    ground_truth_labels = []

    # Check if the cell_type_field exists in adata.obs
    if cell_type_field not in adata.obs:
        typer.echo(
            f"Warning: '{cell_type_field}' not found in adata.obs. Using -1 as ground truth labels."
        )
        has_ground_truth = False
    else:
        has_ground_truth = True

    # Process in batches
    with tqdm(total=num_cells) as pbar:
        # Create a zero inflated data batch. We assume that each sample has the same
        # genes so we don't need to re-allocate this everytime, just inflate into it.
        inflated_batch = np.zeros((batch_size, model_input_size), dtype=np.float32)

        for batch_start in range(0, num_cells, batch_size):
            batch_end = min(batch_start + batch_size, num_cells)
            batch_size_actual = batch_end - batch_start

            # Get batch of expression data
            batch_expression = adata.X[batch_start:batch_end]

            # Handle last batch if its smaller then batch_size
            if batch_size_actual < batch_size:
                inflated_batch = np.zeros(
                    (batch_size_actual, model_input_size), dtype=np.float32
                )

            # # Convert to dense if it's sparse
            # if isinstance(batch_expression, np.ndarray) == False:
            #     batch_expression = batch_expression.toarray()

            # Fill in the data using the inflation map
            for sample_idx, model_idx in inflation_map.items():
                inflated_batch[:, model_idx] = batch_expression[:, sample_idx]

            # Run the model
            model_input = {"input": inflated_batch.astype(np.float32)}
            batch_predictions, batch_encodings = model_session.run(
                ["topk_indices", "encoding"], model_input
            )

            # Store the encodings
            encodings.append(batch_encodings)

            # Store just the top prediction indice
            predictions.append(batch_predictions[:, 0])

            # Extract ground truth class labels
            if has_ground_truth:
                batch_labels = adata.obs[cell_type_field].values[batch_start:batch_end]
                # Convert class names to indices
                batch_label_indices = np.array(
                    [class_to_idx.get(label, -1) for label in batch_labels],
                    dtype=np.int32,
                )
            else:
                # If no ground truth available, use -1 as placeholder
                batch_label_indices = np.full(batch_size_actual, -1, dtype=np.int32)

            ground_truth_labels.append(batch_label_indices)

            pbar.update(batch_size_actual)

    # Combine all batches
    all_encodings = np.vstack(encodings)
    all_predictions = np.concatenate(predictions)
    all_ground_truth = np.concatenate(ground_truth_labels)

    # Save the encodings
    encodings_path = onnx_model_path.with_name(f"{onnx_model_path.stem}-encodings.npy")
    np.save(encodings_path, all_encodings)

    # Save the predictions
    predictions_path = onnx_model_path.with_name(
        f"{onnx_model_path.stem}-predictions.npy"
    )
    np.save(predictions_path, all_predictions)

    # Create array of (ground_truth, prediction) pairs
    label_pairs = np.column_stack((all_ground_truth, all_predictions))

    # Save as binary file
    labels_bin_path = onnx_model_path.with_name(f"{onnx_model_path.stem}-labels.bin")
    label_pairs.astype(np.int32).flatten().tofile(labels_bin_path)

    typer.echo(f"Saved encodings to {encodings_path}")
    typer.echo(f"Saved predictions to {predictions_path}")
    typer.echo(f"Saved label pairs (ground truth, prediction) to {labels_bin_path}")


def create_inflation_map(adata, encoder_model_path: str) -> Dict[int, int]:
    """
    Create a mapping between the model's expected gene indices and the sample's gene indices.

    Args:
        adata: AnnData sample
        encoder_model_path: Path to the encoder model

    Returns:
        Dictionary mapping from sample gene indices to model gene indices
    """
    # Load the model genes
    model_genes_path = Path(encoder_model_path).with_suffix(".genes")
    with open(model_genes_path, "r") as f:
        model_genes = [line.strip() for line in f]

    # Create a mapping from gene names to indices for the model
    model_gene_to_idx = {gene: idx for idx, gene in enumerate(model_genes)}

    # Get the sample genes
    sample_genes = adata.var_names.tolist()

    # Create the inflation map: sample index -> model index
    inflation_map = {}

    # For each gene in the sample, find its index in the model
    for sample_idx, gene_name in enumerate(sample_genes):
        if gene_name in model_gene_to_idx:
            # If the gene exists in the model, add it to the mapping
            model_idx = model_gene_to_idx[gene_name]
            inflation_map[sample_idx] = model_idx

    print(
        f"{adata.X.shape[1] - len(inflation_map)} genes in the sample and not in the model"
    )
    return inflation_map


@app.command()
def train(
    sample_encodings_path: Path = typer.Argument(
        ..., help="Path to encodings.npy file"
    ),
    num_encodings: Optional[int] = typer.Option(
        None,
        help="Limit the total number of encodings used for training, use all if None",
    ),
) -> None:
    """
    Train a parametric umap dimensionality reduction model on the encodings.

    Args:
        sample_encodings_path: Path to sample-encodings.npy file
        num_encodings: Limit the total number of encodings used for training, use all if None
    """
    typer.echo(f"Training parametric UMAP model from encodings {sample_encodings_path}")

    # Load the encodings
    encodings = np.load(sample_encodings_path)

    # Determine number of encodings to use
    num_encodings = num_encodings if num_encodings is not None else encodings.shape[0]
    if num_encodings is not None:
        num_encodings = min(encodings.shape[0], num_encodings)

    encodings = encodings[:num_encodings]

    # Create and fit the model
    model = PUMAP(
        encoder=None,  # nn.Module, None for default
        decoder=None,  # nn.Module, True for default, None for encoder only
        n_neighbors=10,
        min_dist=0.1,
        metric="euclidean",
        n_components=2,
        beta=1.0,  # How much to weigh reconstruction loss for decoder
        reconstruction_loss=torch.nn.functional.binary_cross_entropy_with_logits,  # pass in custom reconstruction loss functions
        random_state=None,
        lr=1e-3,
        epochs=10,
        num_workers=8,
        num_gpus=1,
        match_nonparametric_umap=False,  # Train network to match embeddings from non parametric umap
    )

    model.fit(torch.from_numpy(encodings))

    # Export to ONNX
    output_path = Path(sample_encodings_path).with_name(
        f"{Path(sample_encodings_path).stem.replace('-encodings', '')}-pumap.onnx"
    )
    torch.onnx.export(
        model.model.encoder.encoder,
        torch.zeros(1, encodings.shape[1]),
        output_path,
        training=torch.onnx.TrainingMode.EVAL,
        input_names=["input"],
        output_names=["output"],
        export_params=True,
        opset_version=12,
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    typer.echo(f"Saved clustering model to {output_path}")

    typer.echo(f"Computing mappings for {sample_encodings_path}...")
    mappings = model.transform(torch.from_numpy(encodings))

    # Save the mappings
    output_path = Path(sample_encodings_path).with_name(
        f"{Path(sample_encodings_path).stem.replace('-encodings', '')}-mappings.npy"
    )
    np.save(output_path, mappings)
    typer.echo(f"Saved mappings to {output_path}")


@app.command()
def map(
    model_path: Path = typer.Argument(..., help="Path to mapper.onnx"),
    sample_encodings_path: Path = typer.Argument(
        ..., help="Path to sample-encodings.npy file"
    ),
    batch_size: int = typer.Option(
        32, help="Number of samples to process in each batch"
    ),
    num_samples: Optional[int] = typer.Option(
        None, help="Limit the total number of inputs processed, process all if None"
    ),
) -> None:
    """
    Map the encodings to 2d coordinates using the mapper onnx model.

    Args:
        model_path: Path to pumap.onnx
        sample_encodings_path: Path to sample-encodings.npy file
        batch_size: Number of samples to process at a time
        num_samples: Limit the total number of inputs processed, process all if None
    """
    typer.echo(f"Mapping samples from {sample_encodings_path} using model {model_path}")

    # Load the cluster model
    onnx_session = ort.InferenceSession(str(model_path))
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name

    input_shape = onnx_session.get_inputs()[0].shape
    output_shape = onnx_session.get_outputs()[0].shape

    # Load the encodings
    encodings = np.load(sample_encodings_path)

    # Determine number of samples to process
    num_encodings = num_samples if num_samples is not None else encodings.shape[0]
    if num_samples is not None:
        num_encodings = min(encodings.shape[0], num_samples)

    typer.echo(f"Processing {num_encodings} encodings with batch size {batch_size}")

    # Initialize numpy array to store mappings
    all_mappings = []

    # Process in batches
    with tqdm(total=num_encodings) as pbar:
        for batch_start in range(0, num_encodings, batch_size):
            batch_end = min(batch_start + batch_size, num_encodings)
            batch_size_actual = batch_end - batch_start

            # Get batch of encodings
            batch_encodings = encodings[batch_start:batch_end]

            # Run the model
            model_input = {input_name: batch_encodings.astype(np.float32)}
            batch_mappings = onnx_session.run([output_name], model_input)[0]

            # Store the mappings
            all_mappings.append(batch_mappings)

            pbar.update(batch_size_actual)

    # Combine all batches
    mappings = np.concatenate(all_mappings)

    # Save the mappings
    output_path = Path(sample_encodings_path).with_name(
        f"{Path(sample_encodings_path).stem.replace('-encodings', '')}-mappings.npy"
    )
    np.save(output_path, mappings)

    # Flatten the array to [x1, y1, x2, y2, ...]
    flat_mappings = mappings.flatten()
    output_path = Path(sample_encodings_path).with_name(
        f"{Path(sample_encodings_path).stem.replace('-encodings', '')}-mappings.bin"
    )
    flat_mappings.tofile(output_path)

    typer.echo(f"Saved mappings to {output_path}")


if __name__ == "__main__":
    app()
