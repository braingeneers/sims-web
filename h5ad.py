import numpy as np
import pandas as pd
import anndata as ad
from scipy.sparse import csr_matrix

print(ad.__version__)

adata = ad.read_h5ad("data/pbmc3k_processed.h5ad")
