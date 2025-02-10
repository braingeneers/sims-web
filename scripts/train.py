from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from scsims import SIMS
import anndata as an
import scanpy as sc

adata = an.read_h5ad("data/pbmc3k_processed.h5ad")  # can read h5 using anndata as well
class_label = "louvain"

adata = an.read_h5ad("data/human_log1p_top10k_genes.h5ad")
class_label = "cell_label"

adata = an.read_h5ad("data/human.h5ad")
# # Load your adata as usual
# # adata = sc.read("human")
# # Normalizing to median total counts
# sc.pp.normalize_total(adata)
# # Logarithmize the data
# sc.pp.log1p(adata)
# sc.pp.highly_variable_genes(adata, n_top_genes=10000, batch_key="cell_label")
# adata = adata[:, adata.var.highly_variable]
# adata.write_h5ad("data/human_log1p_top10k_genes.h5ad")


# adata = an.read_h5ad(
#     "data/rnh027_log1p_only_with_UMAP_and_Organoid_labels.h5ad"
# )  # can read h5 using anndata as well
# class_label = "cell_pred"

sims = SIMS(data=adata, class_label=class_label)
sims.setup_model(
    n_a=64, n_d=64, weights=sims.weights
)  # weighting loss inversely proportional by label freq, helps learn rare cell types (recommended)
sims.setup_trainer(
    accelerator="cpu",
    callbacks=[
        EarlyStopping(
            monitor="val_loss",
            patience=50,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ],
    max_epochs=10,
)
sims.train(num_workers=4)  # num_workers=0 for windows, num_workers=4 for linux
