import scanpy as sc
import numpy as np
import anndata as an
from scsims import SIMS

dataset = sc.datasets.pbmc3k_processed()
sims = SIMS(data=dataset, class_label="louvain")
sims.setup_trainer(accelerator="cpu", devices=1, max_epochs=10)
sims.train()