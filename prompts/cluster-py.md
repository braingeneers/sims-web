Collection of prompts used to generate code for this project

## cluster.py

Create a python script in scripts/cluster.py to encode and cluster single cell expression data

### Coding Instructions

- Use the Typer library to structure the command line tools
- Where possible and appopriate use pytorch for computation
- Use adata library to access .h5ad files
- Use type hints consistently
- Optimize for readability over premature optimization
- Write modular code, using separate files for models, data loading, training, and evaluation
- Follow PEP8 style guide for Python code

### Global parameters:

- batch_size (default to 32) to be used in all commands to adjust how many samples are processed at a time in a batch
- num_samples integer (default to 100) used to limit the total number of inputs processed in all commands so we can run end to end in a reasonable amount of time while developing.

### Commands:

#### encode

Generate encodings for a sample .h5ad file containing single cell expression

##### Parameters:

- encoder_model_path: path to an encoder.onnx model file that will be used to generate encodings
- sample_path: path to a sample.h5ad file (contains the gene expression data) that we'll generate encodings for

##### Functionality

- Instantiate the encoder.onnx model using the onnxruntime
- Open the sample.h5ad file and extract the list of genes from sample.genes (one per line, same root file name as .h5ad but with .genes suffix)
- Compute an 'inflation' mapping of the genes in the model v.s the genes in the sample and leaving genes in the model and not in the sample as zero expression
- Load a batch_size of samples from the sample.h5ad file and inflate their expression using the inflation mapping
- Generate encodings for the batch of samples by passing this inflated expression into the encoder collecting the encodings output from the model
- Write the collected encodings to a sample-encodings.npy file

#### train

Train an unsupervised clustering model

##### Parameters:

- sample_encodings_path: path to a sample-encodings.npy file that contains the encodings for all the samples from sample.h5ad above
- cluster_model_path: path to cluster.onnx which we'll output

##### Functionality

- Load the encodings from the sample-encodings.npy file
- Run the hdbscan library in the forward direction to train it on the encodings. It should be wrapped in a PyTorch nn.Module so later it can be exported to ONNX. See this Grok Chat for details: [Grok Unsupervised Clustering Chat](https://grok.com/chat/20bfaf6e-953d-4f1d-ac0a-ae25e68a8a2d)
- Save the model as to cluster_model_path

#### cluster

Cluster a sample using the unsupervised clustering model

##### Parameters:

- cluster_model_path: path to cluster.onnx
- sample_encodings_path: path to a sample-encodings.npy file that contains the encodings for all the samples from sample.h5ad above

##### Functionality

- Instantiate the cluster.onnx model using the onnxruntime
- Load the encodings from the sample-encodings.npy file
- Instantiate cluster.onnx and run in the forward direction to cluster the encodings
- Write the cluster labels to a sample-cluster-labels.npy file
