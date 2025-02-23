/**
 * Browser Web Worker that runs an ONNX model on an h5ad file and stores the
 * results in IndexedDB. These results include:
 * - Cell type predictions with probabilities
 * - Ranked genes per class that explain the prediction (based on attention)
 * - UMAP coordinates for each cell
 *
 * To start prediction, the worker expects a message with the following data:
 * {
 *   type: "startPrediction",
 *   modelID: "model_id",
 *   modelURL: "model_url",
 *   h5File: File,
 *   cellRangePercent: 100
 * }
 *
 * The worker will then download the model, h5ad file, and run the prediction
 * and store the results in IndexedDB. It will send "predictionProgress" messages to the
 * main thread to update the progress bar and "finishedPrediction" message when done.
 * If an error occurs, it will send an "predictionError" message with the error.
 *
 * The results are stored in IndexedDB with the following schema:
 * {
 *   datasetLabel: "dataset_label",
 *   coords: Float32Array, // 2D UMAP coordinates
 *   cellTypeNames: Array, // Array of cell type names
 *   cellTypes: Array, // Array of cell type indices
 *   modelID: "model_id",
 *   topGeneIndicesByClass: Array, // Array of top gene indices per class
 *   genes: Array, // Array of gene names
 *   overallTopGenes: Array, // Array of top gene indices overall
 *   cellNames: Array, // Array of cell names
 *   predictions: Array, // Array of cell type predictions
 *   probabilities: Array // Array of prediction probabilities
 * }
 *
 * The h5 file is read using h5wasm which is a WebAssembly of the h5 library.
 * We utilize its ability to map the file into the browsers file system and
 * thereby read the gene expression data incrementally to support unlimited
 * file sizes. Note this requires the expression data to be stores in
 * column-major order - which most h5ad files are.
 *
 * The prediction is run in multiple threads fed by filling double buffers
 * from the h5 file in a separate thread towards keeping all the threads busy.
 */
import h5wasm from "h5wasm";
import Prando from "prando";
import * as UMAP from "umap-js";

import * as ort from "onnxruntime-web";

import { openDB } from "idb";

// Dictionary with various model information (id, genes, session)
self.model = null;

// Top number of genes to return to explain per class (i.e. Top K)
self.numExplainGenes = 10;

// #classes x #genes matrix to accumulate attention for explaining predictions
self.attentionAccumulators = null;

// Number of threads to use for inference. Use all but one for the GUI to run in
self.numThreads = navigator.hardwareConcurrency - 1;

// Tuned batch size - if I/O, pre-processing and inflation is fast relative to
// the model inference then increase the batch size so that inference is never
// waiting on data. If the model is very large and inference is slow then
// reduce the batch size so that inference can be parallelized across more
// threads. The ONNX model supports variable size batches which plays into
// this as well.
self.batchSize = self.numThreads - 1;

console.log(`Number of threads: ${self.numThreads}`);
console.log(`Batch size: ${self.batchSize}`);

// Limit how many UMAP points we calculate which limits memory by limiting the
// encoding vectors we keep around
self.maxNumCellsToUMAP = 2000;

// Handle messages from the main thread
self.addEventListener("message", async function (event) {
  if (event.data.type === "startPrediction") {
    predict(
      event.data.modelID,
      event.data.modelURL,
      event.data.h5File,
      event.data.cellRangePercent
    );
  }
});

/**
 * Create an ONNX Runtime session for the selected model
 * @param {string} modelURL - The URL of the model
 * @param {string} id - The id of the model to load
 * @returns {Promise} - A promise that resolves to a model session dictionary
 */
async function instantiateModel(modelURL, modelID) {
  console.log(`Instantiating model ${modelID} from ${modelURL}`);
  self.postMessage({ type: "status", message: "Downloading model..." });

  // Fetch the model gene list
  let response = await fetch(`${modelURL}/${modelID}.genes`);
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  const genes = (await response.text()).split("\n");
  console.log("Model Genes", genes.slice(0, 5));

  // Fetch the model classes
  response = await fetch(`${modelURL}/${modelID}.classes`);
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  const classes = (await response.text()).split("\n");
  console.log("Model Classes", classes);

  // Fetch the model ONNX file incrementally to show progress
  response = await fetch(`${modelURL}/${modelID}.onnx`);
  if (!response.ok) {
    throw new Error(`Error fetching onnx file: ${response.status}`);
  }
  const contentLength = response.headers.get("content-length");
  if (!contentLength) {
    throw new Error("Content-Length header is missing");
  }
  const totalBytes = parseInt(contentLength, 10);
  let loadedBytes = 0;

  // Read the response body as a stream
  const reader = response.body.getReader();
  const chunks = [];

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loadedBytes += value.length;

    // Send progress update to the main thread
    self.postMessage({
      type: "predictionProgress",
      message: "Downloading model...",
      countFinished: loadedBytes,
      totalToProcess: totalBytes,
    });
  }

  // Combine all chunks into a single ArrayBuffer
  let modelArray = new Uint8Array(loadedBytes);
  let position = 0;
  for (let chunk of chunks) {
    modelArray.set(chunk, position);
    position += chunk.length;
  }

  // Initialize ONNX Runtime environment
  self.postMessage({ type: "status", message: "Instantiating model..." });
  // See https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html
  ort.env.wasm.numThreads = self.numThreads;
  ort.env.wasm.proxy = true;
  let options = {
    executionProviders: ["wasm"], // alias of 'cpu'
    executionMode: "parallel",
  };

  // Create the InferenceSession with the model ArrayBuffer we fetched incrementally
  const session = await ort.InferenceSession.create(modelArray.buffer, options);
  console.log("Model Output names", session.outputNames);

  return { modelID, session, genes, classes };
}

/*
 * Precompute the inflation indices for the sample gene list
 * @param {Array} currentModelGenes - The gene list of the model
 * @param {Array} sampleGenes - The gene list of the sample
 * @returns {Array} - The inflation indices
 * These are used in fillBatchData inflate each sample from sample gene list space
 * into the model's gene list space
 */
function precomputeInflationIndices(currentModelGenes, sampleGenes) {
  let inflationIndices = [];
  for (let geneIndex = 0; geneIndex < sampleGenes.length; geneIndex++) {
    inflationIndices.push(currentModelGenes.indexOf(sampleGenes[geneIndex]));
  }
  const missingGenesInModel = inflationIndices.filter((x) => x === -1).length;
  console.log(`Missing genes in model: ${missingGenesInModel}`);
  return inflationIndices;
}

/**
 * Fill the batch data and inflate it into the model's gene list space
 * @param {number} batchStart - The start index of the batch
 * @param {number} currentBatchSize - The size of the batch
 * @param {Array} data - The data array
 * @param {Array} indices - The indices array
 * @param {Array} indptr - The indptr array
 * @param {boolean} isSparse - Whether the data is sparse
 * @param {Array} sampleGenes - The gene list of the sample
 * @param {Array} inflationIndices - The inflation indices
 * @param {Array} inflatedBatchData - The inflated batch data
 * This function fills the batch data and inflates it into the model's gene list space
 * in one step. It also handles both sparse and non-sparse data.
 */
function fillBatchData(
  batchStart,
  currentBatchSize,
  data,
  indices,
  indptr,
  isSparse,
  sampleGenes,
  inflationIndices,
  inflatedBatchData
) {
  // Fill batchData and inflate in one step
  for (let batchSlot = 0; batchSlot < currentBatchSize; batchSlot++) {
    const cellIndex = batchStart + batchSlot;
    const batchOffset = batchSlot * self.model.genes.length;

    if (isSparse) {
      // Sparse data stored column major
      const [start, end] = indptr.slice([[cellIndex, cellIndex + 2]]);
      const values = data.slice([[start, end]]);
      const valueIndices = indices.slice([[start, end]]);

      for (let j = 0; j < valueIndices.length; j++) {
        const sampleIndex = inflationIndices[valueIndices[j]];
        if (sampleIndex !== -1) {
          inflatedBatchData[batchOffset + sampleIndex] = values[j];
        }
      }
    } else {
      // Non-sparse stored column major
      // Load up an intermediate buffer with h5wasm slice so we don't
      // call into h5wasm for every value
      let sampleExpression = null;
      if (data.shape.length === 1) {
        // Direct 1D dense array mapping
        sampleExpression = data.slice([
          [
            cellIndex * sampleGenes.length,
            (cellIndex + 1) * sampleGenes.length,
          ],
        ]);
      } else if (data.shape.length === 2) {
        // Direct 2D matrix mapping
        sampleExpression = data.slice([
          [cellIndex, cellIndex + 1],
          [0, sampleGenes.length],
        ]);
      } else {
        throw new Error("Unsupported data shape");
      }
      for (let geneIndex = 0; geneIndex < sampleGenes.length; geneIndex++) {
        const sampleIndex = inflationIndices[geneIndex];
        if (sampleIndex !== -1) {
          inflatedBatchData[batchOffset + sampleIndex] =
            sampleExpression[geneIndex];
        }
      }
    }
  }
}

/**
 * Run the prediction on the model and store the results in IndexedDB
 * @param {string} modelID - The id of the model
 * @param {string} modelURL - The URL of the model
 * @param {File} h5File - The h5ad file
 * @param {number} cellRangePercent - The percentage of cells to process
 * When finished, it sends a "finishedPrediction" message to the main thread.
 */
async function predict(modelID, modelURL, h5File, cellRangePercent) {
  self.postMessage({ type: "status", message: "Loading libraries..." });
  const Module = await h5wasm.ready;
  const { FS } = Module;
  console.log("h5wasm loaded");

  try {
    // Load the model if it's not already loaded
    if (!self.model || self.model.id !== modelID) {
      self.model = await instantiateModel(modelURL, modelID);
    }

    // Reset attention accumulators
    self.attentionAccumulators = new Float32Array(
      self.model.classes.length * self.model.genes.length
    );

    // Load the h5ad file mapping it to the /work directory so we can read
    // it with h5wasm incrementally to support unlimited file sizes
    // We also figure out the list of genes in the sample and the list of cell names
    self.postMessage({ type: "status", message: "Loading file" });
    if (!FS.analyzePath("/work").exists) {
      FS.mkdir("/work");
    }
    FS.mount(FS.filesystems.WORKERFS, { files: [h5File] }, "/work");

    const annData = new h5wasm.File(`/work/${h5File.name}`, "r");
    console.log(annData);

    console.log(`Top level keys: ${annData.keys()}`);

    let cellNames = [];
    if (annData.get("obs").type == "Dataset") {
      cellNames = annData.get("obs").value.map((e) => e[0]);
    } else if (annData.get("obs").type == "Group") {
      if (annData.get("obs").keys().includes("index")) {
        cellNames = annData.get("obs/index").value;
      } else if (annData.get("obs").keys().includes("_index")) {
        cellNames = annData.get("obs/_index").value;
      } else {
        throw new Error("Could not find cell names");
      }
    } else {
      throw new Error("Could not find cell names");
    }

    let sampleGenes = [];
    if (annData.get("var").type == "Dataset") {
      sampleGenes = annData.get("var").value.map((e) => e[0]);
    } else if (annData.get("var").type == "Group") {
      if (annData.get("var").keys().includes("index")) {
        sampleGenes = annData.get("var/index").value;
      } else if (annData.get("var").keys().includes("_index")) {
        sampleGenes = annData.get("var/_index").value;
      } else {
        throw new Error("Could not find genes");
      }
    } else {
      throw new Error("Could not find genes");
    }

    const totalNumCells = cellNames.length;

    // Limit the number of cells to process based on % slider
    cellNames = cellNames.slice(0, (cellRangePercent * cellNames.length) / 100);

    // Setup all the data structures for reading the expression data incrementally
    // including dealing with sparse and inflating the sample gene list into the
    // model's gene list.
    let isSparse = false;
    let data = null;
    let indices = null;
    let indptr = null;
    if (annData.get("X").type == "Dataset") {
      isSparse = false;
      data = annData.get("X");
    } else if (annData.get("X").type == "Group") {
      isSparse = true;
      data = annData.get("X/data");
      indices = annData.get("X/indices");
      indptr = annData.get("X/indptr");
    }

    const labels = [];
    const encodings = [];
    const inflationIndices = precomputeInflationIndices(
      self.model.genes,
      sampleGenes
    );

    const startTime = Date.now(); // Record start time

    // Initialize double buffers of batches
    const buffers = [
      {
        size: 0,
        data: new Float32Array(
          Math.min(self.batchSize, cellNames.length) * self.model.genes.length
        ),
      },
      {
        size: 0,
        data: new Float32Array(
          Math.min(self.batchSize, cellNames.length) * self.model.genes.length
        ),
      },
    ];
    let activeBuffer = 0;

    // Fill the first buffer to kickstart the process whereby while prediction runs
    // on the first buffer, the second buffer is filled with
    // the next batch of cells.
    buffers[activeBuffer].size = Math.min(self.batchSize, cellNames.length);
    fillBatchData(
      0,
      buffers[activeBuffer].size,
      data,
      indices,
      indptr,
      isSparse,
      sampleGenes,
      inflationIndices,
      buffers[activeBuffer].data
    );

    // Begin processing batches of cells double buffer style
    for (
      let batchStart = 0;
      batchStart < cellNames.length;
      batchStart += self.batchSize
    ) {
      // Start inference async on the active buffer
      const inputTensor = new ort.Tensor(
        "float32",
        buffers[activeBuffer].data,
        [buffers[activeBuffer].size, self.model.genes.length]
      );
      const inferencePromise = self.model.session.run({ input: inputTensor });

      // Fill next buffer while inference runs asynchronously
      const nextBuffer = (activeBuffer + 1) % 2;
      const nextStart = batchStart + self.batchSize;
      if (nextStart < cellNames.length) {
        const nextEnd = Math.min(nextStart + self.batchSize, cellNames.length);
        const nextSize = nextEnd - nextStart;
        buffers[nextBuffer].size = Math.min(
          nextSize,
          cellNames.length - nextStart
        );
        if (nextSize < Math.min(self.batchSize, cellNames.length)) {
          // On the last batch and its less then full size so we need to
          // resize the Float32Array for the ort.Tensor creator
          buffers[nextBuffer].data = new Float32Array(
            nextSize * self.model.genes.length
          );
        }
        fillBatchData(
          nextStart,
          buffers[nextBuffer].size,
          data,
          indices,
          indptr,
          isSparse,
          sampleGenes,
          inflationIndices,
          buffers[nextBuffer].data
        );
      }

      // Wait for inference to complete on the current buffer
      const output = await inferencePromise;

      // Parse and store results for each cell in the batch
      for (
        let batchSlot = 0;
        batchSlot < buffers[activeBuffer].size;
        batchSlot++
      ) {
        const overallCellIndex = batchStart + batchSlot;
        labels.push([
          Array.from(
            output.topk_indices.data.slice(batchSlot * 3, batchSlot * 3 + 3)
          ).map(Number),
          output.probs.data.slice(batchSlot * 3, batchSlot * 3 + 3),
        ]);

        // Only push up to maxNumCellsToUMAP so we limit the memory consumption as these
        // are 32 float vectors per cell
        if (overallCellIndex < self.maxNumCellsToUMAP) {
          // Each encoding row is shaped by your model: e.g. 16 dims
          const encSize = output.encoding.dims[1];
          const encSliceStart = batchSlot * encSize;
          const encSliceEnd = encSliceStart + encSize;
          encodings.push(
            output.encoding.data.slice(encSliceStart, encSliceEnd)
          );
        }

        // Add attention into the predicted class accumulator
        for (let i = 0; i < self.model.genes.length; i++) {
          const classIndex = labels[labels.length - 1][0][0];
          self.attentionAccumulators[
            classIndex * self.model.genes.length + i
          ] += output.attention.data[batchSlot * self.model.genes.length + i];
        }
      }

      // Post progress update
      self.postMessage({
        type: "predictionProgress",
        message: `Predicting ${cellNames.length} out of ${totalNumCells}...`,
        countFinished: nextStart,
        totalToProcess: cellNames.length,
      });

      // Swap buffers
      activeBuffer = nextBuffer;
    }

    // All done so unmount the h5 file from the browsers file system
    annData.close();
    FS.unmount("/work");

    // Record end time and calculate elapsed time of prediction only
    const endTime = Date.now(); // Record end time
    const elapsedTime = (endTime - startTime) / 60000; // Calculate elapsed time in minutes

    // ========================================================================
    // Run UMAP on the encoding to calculate a 2D projection
    // ========================================================================
    const prando = new Prando(42);
    const random = () => prando.next();

    const umap = new UMAP.UMAP({
      random,
      nComponents: 2,
      nEpochs: 400,
      nNeighbors: 15,
    });

    let coordinates = null;
    try {
      coordinates = await umap.fitAsync(encodings, (epochNumber) => {
        // check progress and give user feedback, or return `false` to stop
        self.postMessage({
          type: "predictionProgress",
          message: `Computing the first ${encodings.length} coordinates...`,
          countFinished: epochNumber,
          totalToProcess: umap.getNEpochs(),
        });
      });
    } catch (error) {
      self.postMessage({ type: "predictionError", error });
      throw error;
    }

    // ========================================================================
    // Calculate top K gene indices per class and overall as well as
    // the overall top k gene indices for all predictions. These are used
    // to explain the predictions per class and overall.
    // ========================================================================
    let topGeneIndicesByClass = [];

    function topKIndices(x, k) {
      const indices = Array.from(x.keys());
      indices.sort((a, b) => x[b] - x[a]);
      return indices.slice(0, k);
    }

    let overallAccumulator = new Float32Array(self.model.genes.length);
    for (let i = 0; i < self.model.classes.length; i++) {
      topGeneIndicesByClass.push(
        topKIndices(
          self.attentionAccumulators.slice(
            i * self.model.genes.length,
            (i + 1) * self.model.genes.length
          ),
          self.numExplainGenes
        )
      );
      for (let j = 0; j < self.model.genes.length; j++) {
        overallAccumulator[j] +=
          self.attentionAccumulators[i * self.model.genes.length + j];
      }
    }
    let overallTopGenes = topKIndices(overallAccumulator, self.numExplainGenes);

    const cellTypes = labels.map((label) => label[0][0]);

    // Store results in IndexedDB
    const db = await openDB("sims-web");
    const tx = db.transaction("datasets", "readwrite");
    const store = tx.objectStore("datasets");
    await store.put({
      // Note: these are expected and used by the UCSC cell browser so don't change
      // without coordinating with the UCSC cell browser team (i.e. Max!)
      datasetLabel: h5File.name,
      coords: coordinates.flat(), // Float32Array of length 2*n
      cellTypeNames: self.model.classes, // Array of strings
      cellTypes: cellTypes,

      // Used for the UIs and export in this application
      modelID: self.model.id,
      topGeneIndicesByClass, // Array of arrays, 1 per class, of top indices
      genes: self.model.genes, // Array of strings
      overallTopGenes,
      cellNames,
      predictions: labels.map((label) => label[0]),
      probabilities: labels.map((label) => label[1]),
    });
    await tx.done;
    db.close();

    // Let the main thread know we're done and results are ready in IndexDB
    self.postMessage({
      type: "finishedPrediction",
      datasetLabel: h5File.name,
      elapsedTime,
      totalProcessed: cellNames.length,
      totalNumCells,
    });
  } catch (error) {
    // FS.unmount("/work");
    self.postMessage({ type: "predictionError", error: error });
  }
}
