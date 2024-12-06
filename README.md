# sims-web
Run [SIMS](https://github.com/braingeneers/SIMS) in the browser using [h5wasm](https://github.com/usnistgov/h5wasm) to read local AnnData (.h5ad) files and [ONNX](https://onnxruntime.ai/) to run the model.

# Demo
[index.html](https://public.gi.ucsc.edu/~rcurrie/sims/) opens an h5ad in the browser and runs a selected SIMs model and displays predictions.

![Alt text](screenshot.png?raw=true "SIMS Web Screenshot")

# Running

Export a SIMS checkpoint to an onnx file and list of genes. Note this assumes you have the SIMS repo as a peer to this one so it can load the model definition.
```
python sims-to-onnx.py models/default.ckpt
```

Check the core model for compatibility with onnx
```
python -m onnxruntime.tools.check_onnx_model_mobile_usability models/default.onnx
```

Serve the web app and models locally
```
python -m http.server 3000
```

# Memory Requirements
[worker.js](worker.js) uses h5wasm slice() to read data from the cell by gene matrix (i.e. X). As these data on disk are typically stored row major (i.e. all data for a cell is contiguous) we can process the sample incrementally keeping memory requirements to a minimum. Reading cell by cell from a 5.3G h5ad file consumed just under 30M of browser memory. YMMV.

# Functional Experiments 

[h5ad.html](h5ad.html) demonstrates reading an h5ad file over http and using h5wasm in the browser and extracting the gene names and expression matrix values. Must be served locally to comply with CORS. An actual implementation would present an open file dialog to read a file locally.

[onnx.html](onnx.html) demonstrates loading the sims.onnx file in the browser and running forward inference on a sample zero tensor.

# References
[Open Neural Network Exchange (ONNX)](https://onnx.ai/)

[ONNX Runtime Web (WASM Backend)](https://onnxruntime.ai/docs/get-started/with-javascript/web.html)

[ONNX Runtime Web Platform Functionality Details](https://www.npmjs.com/package/onnxruntime-web)

[ONNX Runtime Javascript Examples](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/js)

[Graphical ONNX Editor](https://github.com/ZhangGe6/onnx-modifier)
[Classify images in a web application with ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/classify-images-nextjs-github-template.html)

[h5wasm](https://github.com/usnistgov/h5wasm)

[anndata/h5ad file structure](https://anndata.readthedocs.io/en/latest/tutorials/notebooks/getting-started.html) and [on disk format](https://anndata.readthedocs.io/en/latest/fileformat-prose.html)

[SIMS Streamlit App](https://sc-sims-app.streamlit.app/) and [Source](https://github.com/jesusgf1/sims_app/blob/main/streamlit_app.py)
