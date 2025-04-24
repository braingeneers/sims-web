# cell-space

sims-web.next()

Refactor to support a labeling and unsupervised clustering workflow

# Adding a model
Given a model.ckpt and pumap.onnx 

# Datasets & Models

## Allen Brain

allen-celltypes+human-cortex+various-cortical-areas.ckpt

Dataset used to train the model
AWS location (curated adata with CellType column): s3://braingeneersdev/jgf/jing_models/allen-human/human-cortex
CellBrowser URL: https://cells.ucsc.edu/?proj=Allen+Brain+Atlas&ds=allen-celltypes+human-cortex

Test dataset to try and embed both datasets together:
Aws location (curated adata with CellType column): s3://braingeneersdev/jgf/jing_models/allen-human/m1-region/
CellBrowser URL: https://cells.ucsc.edu/?proj=Allen+Brain+Atlas&ds=allen-celltypes+human-cortex
