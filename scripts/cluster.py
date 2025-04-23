#!/usr/bin/env python
"""
Single cell expression data encoder and clustering script.

This script provides functionality to encode and cluster single cell expression data
using ONNX models with PyTorch computation backend.
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Union

import numpy as np
import torch
import typer
import onnxruntime as ort
import hdbscan
import anndata as ad
from torch import nn
from tqdm import tqdm

# Create Typer app
app = typer.Typer(help="Single cell expression data encoder and clustering")


@app.command()
def predict(
    onnx_model_path: Path = typer.Argument(..., help="Path to model .onnx file"),
    sample_path: Path = typer.Argument(..., help="Path to sample.h5ad file"),
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

    # Load the sample data
    adata = ad.read_h5ad(sample_path)

    # Create an inflation mapping between model genes and sample genes
    inflation_map = create_inflation_map(adata, onnx_model_path)

    # Determine number of cells to process
    num_cells = num_samples if num_samples is not None else adata.n_obs
    if num_samples is not None:
        num_cells = min(adata.n_obs, num_samples)

    typer.echo(f"Processing {num_cells} cells with batch size {batch_size}")

    # Initialize numpy array to store encodings
    predictions = []
    encodings = []

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

            pbar.update(batch_size_actual)

    # Combine all batches
    all_encodings = np.vstack(encodings)
    all_predictions = np.concatenate(predictions)

    # Save the encodings
    encodings_path = onnx_model_path.with_name(f"{onnx_model_path.stem}-encodings.npy")
    np.save(encodings_path, all_encodings)

    # Save the predictions
    predictions_path = onnx_model_path.with_name(
        f"{onnx_model_path.stem}-predictions.npy"
    )
    np.save(predictions_path, all_predictions)

    typer.echo(f"Saved encodings to {encodings_path}")
    typer.echo(f"Saved predictions to {predictions_path}")


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


class HDBSCANCluster(nn.Module):
    """
    Generated by Grok in this chat: https://grok.com/chat/20bfaf6e-953d-4f1d-ac0a-ae25e68a8a2d
    """

    def __init__(
        self, min_cluster_size=5, min_samples=None, cluster_selection_epsilon=0.0
    ):
        """
        HDBSCAN clustering as a PyTorch module

        Args:
            min_cluster_size (int): Minimum size of clusters
            min_samples (int, optional): Minimum number of samples in neighborhood for a point to be a core point
            cluster_selection_epsilon (float): Distance threshold for cluster selection
        """
        super(HDBSCANCluster, self).__init__()
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples if min_samples is not None else min_cluster_size
        self.cluster_selection_epsilon = cluster_selection_epsilon

        # Register buffer for storing cluster labels (will be computed during first forward pass)
        self.register_buffer("labels", torch.tensor([]))
        self.is_fitted = False

    def forward(self, x):
        """
        Forward pass that performs clustering on 32d encodings

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 32)

        Returns:
            torch.Tensor: Cluster labels for each sample (batch_size,)
        """
        if not self.is_fitted:
            # Perform HDBSCAN clustering
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                cluster_selection_epsilon=self.cluster_selection_epsilon,
                gen_min_span_tree=True,
            )
            labels = clusterer.fit_predict(x)

            # Store labels as buffer
            self.labels = torch.from_numpy(labels)
            self.is_fitted = True

        return self.labels

    def reset(self):
        """Reset the fitted state"""
        self.is_fitted = False
        self.labels = torch.tensor([])


@app.command()
def train_hdbscan(
    sample_encodings_path: Path = typer.Argument(
        ..., help="Path to encodings.npy file"
    ),
    cluster_model_path: Path = typer.Argument(
        ..., help="Path output cluster.onnx and labels.npy file"
    ),
    num_encodings: Optional[int] = typer.Option(
        None,
        help="Limit the total number of encodings used for training, use all if None",
    ),
    min_cluster_size: int = typer.Option(
        5, help="Minimum number of samples for a cluster"
    ),
) -> None:
    """
    Train an unsupervised clustering model using HDBSCAN.

    Args:
        sample_encodings_path: Path to sample-encodings.npy file
        cluster_model_path: Path to directory to output cluster.onnx and labels.npy file
        num_encodings: Limit the total number of encodings used for training, use all if None
        min_cluster_size: Minimum number of samples for a cluster
    """
    typer.echo(
        f"Training HDBSCAN clustering model from encodings {sample_encodings_path}"
    )

    # Load the encodings
    encodings = np.load(sample_encodings_path)

    # Determine number of encodings to use
    num_encodings = num_encodings if num_encodings is not None else encodings.shape[0]
    if num_encodings is not None:
        num_encodings = min(encodings.shape[0], num_encodings)

    encodings = encodings[:num_encodings]

    # Create and fit the model
    model = HDBSCANCluster(min_cluster_size=min_cluster_size)
    labels = model(encodings)

    # Save the labels
    output_path = (
        f"{cluster_model_path}/{Path(sample_encodings_path).stem}-hdbscan-labels.npy"
    )
    np.save(output_path, labels)
    typer.echo(f"Saved labels to {output_path}")

    # Export to ONNX
    output_path = (
        f"{cluster_model_path}/{Path(sample_encodings_path).stem}-hdbscan-labels.onnx"
    )
    torch.onnx.export(
        model,
        torch.zeros(1, 32),
        output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=12,  # Only version that works in onnx webruntime for now...
    )
    typer.echo(f"Saved clustering model to {output_path}")

    # Print results
    unique_labels = torch.unique(labels)
    typer.echo(f"Found {len(unique_labels)} clusters (including noise)")
    for label in unique_labels:
        if label == -1:
            typer.echo(f"Noise points: {(labels == label).sum().item()}")
        else:
            typer.echo(
                f"Cluster {label.item()}: {(labels == label).sum().item()} points"
            )


@app.command()
def cluster(
    cluster_model_path: Path = typer.Argument(..., help="Path to cluster.onnx"),
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
    Cluster a sample using the unsupervised clustering model.

    Args:
        cluster_model_path: Path to cluster.onnx
        sample_encodings_path: Path to sample-encodings.npy file
        batch_size: Number of samples to process at a time
        num_samples: Limit the total number of inputs processed, process all if None
    """
    typer.echo(
        f"Clustering samples from {sample_encodings_path} using model {cluster_model_path}"
    )

    # Load the cluster model
    cluster_session = ort.InferenceSession(str(cluster_model_path))
    input_name = cluster_session.get_inputs()[0].name
    output_name = cluster_session.get_outputs()[0].name

    # Load the encodings
    encodings = np.load(sample_encodings_path)

    # Determine number of samples to process
    num_encodings = num_samples if num_samples is not None else encodings.shape[0]
    if num_samples is not None:
        num_encodings = min(encodings.shape[0], num_samples)

    typer.echo(f"Processing {num_encodings} encodings with batch size {batch_size}")

    # Initialize numpy array to store cluster labels
    all_labels = []

    # Process in batches
    with tqdm(total=num_encodings) as pbar:
        for batch_start in range(0, num_encodings, batch_size):
            batch_end = min(batch_start + batch_size, num_encodings)
            batch_size_actual = batch_end - batch_start

            # Get batch of encodings
            batch_encodings = encodings[batch_start:batch_end]

            # Run the model
            model_input = {input_name: batch_encodings.astype(np.float32)}
            batch_labels = cluster_session.run([output_name], model_input)[0]

            # Store the labels
            all_labels.append(batch_labels)

            pbar.update(batch_size_actual)

    # Combine all batches
    cluster_labels = np.concatenate(all_labels)

    # Save the cluster labels
    output_path = Path(sample_encodings_path).with_name(
        f"{Path(sample_encodings_path).stem.replace('-encodings', '')}-hdbscan-labels.npy"
    )
    np.save(output_path, cluster_labels)

    typer.echo(f"Saved cluster labels to {output_path}")

    # Print some statistics
    unique_labels = np.unique(cluster_labels)
    num_clusters = len(unique_labels[unique_labels >= 0])  # Exclude noise points (-1)
    noise_points = np.sum(cluster_labels == -1)

    typer.echo(f"Found {num_clusters} clusters with {noise_points} noise points")


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
