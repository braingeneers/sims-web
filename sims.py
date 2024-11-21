import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import torch
import onnxruntime
import onnx
import gc

import sys

sys.path.insert(0, "../SIMS")
from scsims import SIMS

import torch.onnx

if __name__ == "__main__":
    # Load the checkpoint
    sims = SIMS(weights_path="models/default.ckpt", map_location=torch.device("cpu"))

    test_tensor = torch.zeros(1, 33694)
    for i in range(100):
        test_tensor[0, i] = 0.5

    # Pytorch
    sims.model.eval()  # This is necessary to avoid dropout
    results = sims.model(test_tensor.float())
    print("sims.model() output")
    print(results[0][0])

    # Try onnx
    session = onnxruntime.InferenceSession("models/default.onnx")
    outputs = session.run(None, {"input.1": test_tensor.numpy()})
    print("onnx session run output")
    print(outputs[0][0])

    # Load the dataset
    adata = sc.read_h5ad("data/pbmc3k.h5ad")
    loader = sims.model._parse_data(adata)
    print("Cell 0 #44 Expression", loader.dataset.data[0, 44])

    sims.model.eval()
    pt_tensor = (
        torch.sparse_coo_tensor(
            loader.dataset.data[0, :].nonzero(),
            loader.dataset.data[0, :].data,
            loader.dataset.data[0, :].shape,
        )
        .to_dense()
        .to(torch.float32)
    )
    results = sims.model(pt_tensor.to_dense())
    print("Cell 0 Raw Predictions:", results[0])
