from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from scsims import SIMS
import anndata as an

adata = an.read_h5ad("data/pbmc3k_processed.h5ad")  # can read h5 using anndata as well
class_label = "louvain"

# adata = an.read_h5ad("data/human_with_labels.h5ad")  # can read h5 using anndata as well
# class_label = "cell_label"

# adata = an.read_h5ad(
#     "data/rnh027_log1p_only_with_UMAP_and_Organoid_labels.h5ad"
# )  # can read h5 using anndata as well
# class_label = "cell_pred"

sims = SIMS(data=adata, class_label=class_label)
sims.setup_model(
    n_a=64, n_d=64, weights=sims.weights
)  # weighting loss inversely proportional by label freq, helps learn rare cell types (recommended)
sims.setup_trainer(
    accelerator="mps",
    callbacks=[
        EarlyStopping(
            monitor="val_loss",
            patience=50,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ],
    max_epochs=1,
)
sims.train()
