# sims-web

Run [SIMS](https://github.com/braingeneers/SIMS) in the browser using [h5wasm](https://github.com/usnistgov/h5wasm) to read local AnnData (.h5ad) files and [ONNX](https://onnxruntime.ai/) to run the model.

# [Demo](https://cells-test.gi.ucsc.edu/sims)

Opens an h5ad in the browser and runs a selected SIMs model and displays predictions.

You can view the default ONNX model via [netron](https://netron.app/?url=https://github.com/braingeneers/sims-web/raw/refs/heads/main/public/models/default.onnx)

![Alt text](screenshot.png?raw=true "SIMS Web Screenshot")

NOTE: This application has not been fully verified as concordant to the python SIMS yet. Currently the predictions are ~90% concordant with SIMS

# Architecture

The front end a single page React web app using Material UI and Vite with no back end - just file storage and an HTTP server is required. The python pieces all relate to converting pytorch models into ONNX and then editing the ONNX graph to move as much of predictions processing into the graph as possible (i.e. LpNorm and SoftMax of probabilities) as well as to expose internal nodes such as the encoder output for clustering and the attention masks for explainability.

# Developing

Install dependencies for the python model exporting and webapp:

```
pip install -r requirements.txt
npm install
```

Export a SIMS checkpoint to an ONNX file and list of genes:

```
python scripts/sims-to-onnx.py checkpoints/default.ckpt public/models/
```

Check a model for compatibility with ONNX:

```
python -m onnxruntime.tools.check_onnx_model_mobile_usability public/models/default.onnx
```

Compare the output of SIMS to ONNX using the python runtime:

```
python scripts/validate.py checkpoints/default.ckpt public/models/default.onnx public/sample.h5ad --decimals 2
```

Serve the web app and exported models locally with auto-reload courtesy of vite:

```
npm run dev
```

Display the compute graph using netron:

```
netron public/models/default.onnx
```

# Memory Requirements

[worker.js](worker.js) uses h5wasm slice() to read data from the cell by gene matrix (i.e. X). As these data on disk are typically stored row major (i.e. all data for a cell is contiguous) we can process the sample incrementally keeping memory requirements to a minimum. Reading cell by cell from a 5.3G h5ad file consumed just under 30M of browser memory. YMMV.

# Performance

ONNX supports multithreaded inference. We allocate total cores - 2 for inference. This leaves 1 thread for the main loop so the UI can remain responsible and 1 thread for ONNX to coordinate via its 'proxy' setting (see worker.js for details).

Processed 1759 cells in 0.18 minutes on a MacBook M3 Pro or around 10k samples per minute.

# Leveraging a GPU

ONNX Web Runtime does have support for GPUs, but unfortunately they don't support all operators yet. Specifically TopK, LpNormalization and GatherElements are not [supported](https://github.com/microsoft/onnxruntime/blob/main/js/web/docs/webgpu-operators.md). See sclblonnx.check(graph) for details.

# References

[Open Neural Network Exchange (ONNX)](https://onnx.ai/)

[ONNX Runtime Web (WASM Backend)](https://onnxruntime.ai/docs/get-started/with-javascript/web.html)

[ONNX Runtime Web Platform Functionality Details](https://www.npmjs.com/package/onnxruntime-web)

[ONNX Runtime Javascript Examples](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/js)

[Alternative Web ONNX Runtime in Rust](https://github.com/webonnx/wonnx)

[ONNX Simplifier](https://github.com/daquexian/onnx-simplifier)

[Netron ONNX Graph Display Website](https://netron.app/)

[Graphical ONNX Editor](https://github.com/ZhangGe6/onnx-modifier)

[Classify images in a web application with ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/classify-images-nextjs-github-template.html)

[h5wasm](https://github.com/usnistgov/h5wasm)

[anndata/h5ad file structure](https://anndata.readthedocs.io/en/latest/tutorials/notebooks/getting-started.html) and [on disk format](https://anndata.readthedocs.io/en/latest/fileformat-prose.html)

[SIMS Streamlit App](https://sc-sims-app.streamlit.app/) and [Source](https://github.com/jesusgf1/sims_app/blob/main/streamlit_app.py)

[TabNet Model for attentive tabular learning](https://youtu.be/g1gMB3v5kzk?si=_7Wx-2giEPea68y8)

[Semi supervised pre training with TabNet](https://www.kaggle.com/code/sisharaneranjana/semi-supervised-pre-training-with-tabnet)

[Self Supervised TabNet](https://www.kaggle.com/code/optimo/selfsupervisedtabnet)

[Classification of Alzheimer's disease using robust TabNet neural networks on genetic data](https://www.aimspress.com/article/doi/10.3934/mbe.2023366)

[Designing interpretable deep learning applications for functional genomics: a quantitative analysis ](https://academic.oup.com/bib/article/25/5/bbae449/7759907)

[Assessing GPT-4 for cell type annotation in single-cell RNA-seq analysis](https://www.biorxiv.org/content/10.1101/2023.04.16.537094v2)
