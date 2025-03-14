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
def encode(
    encoder_model_path: Path = typer.Argument(..., help="Path to encoder.onnx model file"),
    sample_path: Path = typer.Argument(..., help="Path to sample.h5ad file"),
    batch_size: int = typer.Option(32, help="Number of samples to process in each batch"),
    num_samples: int = typer.Option(100, help="Limit the total number of inputs processed"),
) -> None:
    """
    Generate encodings for a sample .h5ad file containing single cell expression data.
    
    Args:
        encoder_model_path: Path to an encoder.onnx model file
        sample_path: Path to a sample.h5ad file containing gene expression data
        batch_size: Number of samples to process at a time
        num_samples: Limit the total number of inputs processed
    """
    typer.echo(f"Encoding samples from {sample_path} using model {encoder_model_path}")
    
    # Instantiate the encoder model
    encoder_session = ort.InferenceSession(str(encoder_model_path))
    
    # # Get model input shape to determine the expected genes
    input_shape = encoder_session.get_inputs()[0].shape
    model_input_size = input_shape[1] if len(input_shape) > 1 else input_shape[0]
    
    # Load the sample data
    adata = ad.read_h5ad(sample_path)
    
    # Create an inflation mapping between model genes and sample genes
    inflation_map = create_inflation_map(adata, encoder_model_path)
    
    # Limit the number of samples
    num_cells = min(adata.n_obs, num_samples)
    typer.echo(f"Processing {num_cells} cells with batch size {batch_size}")
    
    # Initialize numpy array to store encodings
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
                inflated_batch = np.zeros((batch_size_actual, model_input_size), dtype=np.float32)

            # # Convert to dense if it's sparse
            # if isinstance(batch_expression, np.ndarray) == False:
            #     batch_expression = batch_expression.toarray()

            # Fill in the data using the inflation map
            for sample_idx, model_idx in inflation_map.items():
                inflated_batch[:, model_idx] = batch_expression[:, sample_idx]

            # Run the model
            model_input = {"input": inflated_batch.astype(np.float32)}
            batch_encodings = encoder_session.run(["encoding"], model_input)[0]

            # Store the encodings
            encodings.append(batch_encodings)

            pbar.update(batch_size_actual)

    # Combine all batches
    all_encodings = np.vstack(encodings)

    # Save the encodings
    output_path = sample_path.with_name(f"{sample_path.stem}-encodings.npy")
    np.save(output_path, all_encodings)

    typer.echo(f"Saved encodings to {output_path}")


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
    model_genes_path = Path(encoder_model_path).with_suffix('.genes')
    with open(model_genes_path, 'r') as f:
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
    
    print(f"{adata.X.shape[1] - len(inflation_map)} genes in the sample and not in the model")
    return inflation_map

class HDBSCANModule(nn.Module):
    """
    PyTorch module wrapping HDBSCAN clustering for ONNX export.
    """
    def __init__(self, min_cluster_size: int = 5, min_samples: int = 1):
        super().__init__()
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            prediction_data=True,
        )
        self.fitted = False
    
    def fit(self, X: np.ndarray) -> None:
        """
        Fit the HDBSCAN model to the data.
        
        Args:
            X: Input data matrix (samples x features)
        """
        self.clusterer.fit(X)
        self.fitted = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the HDBSCAN model.
        
        Args:
            x: Input tensor (samples x features)
            
        Returns:
            Tensor of cluster labels
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before running forward pass")
        
        # Convert to numpy for HDBSCAN
        x_np = x.detach().cpu().numpy()
        
        # Predict clusters
        labels, _ = hdbscan.approximate_predict(self.clusterer, x_np)
        
        # Convert back to tensor
        return torch.tensor(labels, dtype=torch.int64)


@app.command()
def train(
    sample_encodings_path: Path = typer.Argument(..., help="Path to sample-encodings.npy file"),
    cluster_model_path: Path = typer.Argument(..., help="Path to output cluster.onnx"),
    min_cluster_size: int = typer.Option(5, help="Minimum number of samples for a cluster"),
    min_samples: int = typer.Option(1, help="Number of samples in a neighborhood for a core point"),
    batch_size: int = typer.Option(32, help="Number of samples to process in each batch"),
    num_samples: int = typer.Option(100, help="Limit the total number of inputs processed"),
) -> None:
    """
    Train an unsupervised clustering model using HDBSCAN.
    
    Args:
        sample_encodings_path: Path to sample-encodings.npy file
        cluster_model_path: Path to output cluster.onnx
        min_cluster_size: Minimum number of samples for a cluster
        min_samples: Number of samples in a neighborhood for a core point
        batch_size: Number of samples to process at a time
        num_samples: Limit the total number of inputs processed
    """
    typer.echo(f"Training clustering model from encodings {sample_encodings_path}")
    
    # Load the encodings
    encodings = np.load(sample_encodings_path)
    
    # Limit the number of samples
    encodings = encodings[:min(encodings.shape[0], num_samples)]
    
    # Create and fit the model
    model = HDBSCANModule(min_cluster_size=min_cluster_size, min_samples=min_samples)
    model.fit(encodings)
    
    # Create dummy input for ONNX export
    dummy_input = torch.tensor(encodings, dtype=torch.float32)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        str(cluster_model_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=12,
    )
    
    typer.echo(f"Saved clustering model to {cluster_model_path}")


@app.command()
def cluster(
    cluster_model_path: Path = typer.Argument(..., help="Path to cluster.onnx"),
    sample_encodings_path: Path = typer.Argument(..., help="Path to sample-encodings.npy file"),
    batch_size: int = typer.Option(32, help="Number of samples to process in each batch"),
    num_samples: int = typer.Option(100, help="Limit the total number of inputs processed"),
) -> None:
    """
    Cluster a sample using the unsupervised clustering model.
    
    Args:
        cluster_model_path: Path to cluster.onnx
        sample_encodings_path: Path to sample-encodings.npy file
        batch_size: Number of samples to process at a time
        num_samples: Limit the total number of inputs processed
    """
    typer.echo(f"Clustering samples from {sample_encodings_path} using model {cluster_model_path}")
    
    # Load the cluster model
    cluster_session = ort.InferenceSession(str(cluster_model_path))
    input_name = cluster_session.get_inputs()[0].name
    output_name = cluster_session.get_outputs()[0].name
    
    # Load the encodings
    encodings = np.load(sample_encodings_path)
    
    # Limit the number of samples
    num_samples_actual = min(encodings.shape[0], num_samples)
    encodings = encodings[:num_samples_actual]
    
    typer.echo(f"Processing {num_samples_actual} samples with batch size {batch_size}")
    
    # Initialize numpy array to store cluster labels
    all_labels = []
    
    # Process in batches
    with tqdm(total=num_samples_actual) as pbar:
        for batch_start in range(0, num_samples_actual, batch_size):
            batch_end = min(batch_start + batch_size, num_samples_actual)
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
        f"{Path(sample_encodings_path).stem.replace('-encodings', '')}-cluster-labels.npy"
    )
    np.save(output_path, cluster_labels)
    
    typer.echo(f"Saved cluster labels to {output_path}")
    
    # Print some statistics
    unique_labels = np.unique(cluster_labels)
    num_clusters = len(unique_labels[unique_labels >= 0])  # Exclude noise points (-1)
    noise_points = np.sum(cluster_labels == -1)
    
    typer.echo(f"Found {num_clusters} clusters with {noise_points} noise points")


if __name__ == "__main__":
    app()
