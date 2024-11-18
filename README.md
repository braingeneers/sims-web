# sims-web
Run [SIMS](https://github.com/braingeneers/SIMS) in the browser using [h5wasm](https://github.com/usnistgov/h5wasm) to read local AnnData (.h5ad) files and [ONNX](https://onnxruntime.ai/) to run the model on CPU or GPU if available.

# Demo
[index.html](https://public.gi.ucsc.edu/~rcurrie/sims/) that opens an h5ad in the browser and prints the raw predictions for the first cell out.

# Running
Export a SIMS checkpoint to an onnx file and list of genes
```
python sims-to-onnx.py models/11A_2organoids.ckpt
```

# Functional Experiments 

[h5ad.html](h5ad.html) demonstrates reading an h5ad file over http and using h5wasm in the browser and extracting the gene names and expression matrix values. Must be served locally to comply with CORS. An actual implimentation would present an open file dialog to read a file locally.

[sims-to-onnx.py](sims-to-onnx.py) exports a SIMS checkpoint as an onnx file along with a text list of the genes it expects as input. Note this assumes you have the SIMS repo as a peer to this one so it can load the model definition.

[onnx.html](onnx.html) demonstrates loading the sims.onnx file in the browser and running forward inference on a sample zero tensor.

[index.html](index.html) puts these functional steps together towards a basic MVP

# References
[h5wasm](https://github.com/usnistgov/h5wasm)

[anndata/h5ad file struction](https://anndata.readthedocs.io/en/latest/tutorials/notebooks/getting-started.html)

[Classify images in a web application with ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/classify-images-nextjs-github-template.html)

[ONNX Runtime Web (WASM Backend)](https://onnxruntime.ai/docs/get-started/with-javascript/web.html)

[ONNX Runtime Web Platform Functionality Details](https://www.npmjs.com/package/onnxruntime-web)