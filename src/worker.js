import h5wasm from "h5wasm";
import Prando from "prando";
import * as UMAP from "umap-js";

import * as ort from "onnxruntime-web";

// Includes WebAssembly backend only
// import * as ort from "onnxruntime-web/wasm";

// Includes WebAssembly single-threaded only, with training support
// import * as ort from "onnxruntime-web/training";

// Top K genes to accumulate attention for
self.K = 10;

// Worker global variables
self.model = null;
self.attentionAccumulators = null;

// Leave one thread for the main thread
self.numThreads = navigator.hardwareConcurrency - 1;
// Leave one thread for onnx to proxy from
self.batchSize = self.numThreads - 1;

console.log(`Number of threads: ${self.numThreads}`);
console.log(`Batch size: ${self.batchSize}`);

// Limit how many UMAP points we calculate which limits memory by limiting the
// encoding vectors we keep around
self.maxNumCoords = 2000;

// Handle messages from the main thread
self.addEventListener("message", async function (event) {
  if (event.data.type === "startPrediction") {
    predict(event);
  }
});

/**
 * Create an ONNX Runtime session for the selected model
 * @param {string} id - The id of the model to load
 * @returns {Promise} - A promise that resolves to a model session dictionary
 */
async function instantiateModel(modelURL, id) {
  console.log(`Instantiating model ${id} from ${modelURL}`);
  self.postMessage({ type: "status", message: "Downloading model..." });

  // Load the model gene list
  let response = await fetch(`${modelURL}/${id}.genes`);
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  const genes = (await response.text()).split("\n");
  console.log("Model Genes", genes.slice(0, 5));

  // Load the model classes
  response = await fetch(`${modelURL}/${id}.classes`);
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  const classes = (await response.text()).split("\n");
  console.log("Model Classes", classes);

  response = await fetch(`${modelURL}/${id}.onnx`);
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
      type: "progress",
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

  self.postMessage({ type: "status", message: "Instantiating model..." });
  // Initialize ONNX Runtime environment
  // See https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html
  // ort.env.wasm.wasmPaths = "/dist/";
  ort.env.wasm.numThreads = self.numThreads;
  ort.env.wasm.proxy = true;
  let options = {
    executionProviders: ["wasm"], // alias of 'cpu'
    // executionMode: "sequential",
    executionMode: "parallel",
    // graphOptimizationLevel: "all",
    // inter_op_num_threads: numThreads,
    // intra_op_num_threads: numThreads,
    // enableCpuMemArena: true,
    // enableMemPattern: true,
    // extra: {
    //   optimization: {
    //     enable_gelu_approximation: "1",
    //   },
    //   session: {
    //     intra_op_num_threads: numThreads,
    //     inter_op_num_threads: numThreads,
    //     disable_prepacking: "1",
    //     use_device_allocator_for_initializers: "1",
    //     use_ort_model_bytes_directly: "1",
    //     use_ort_model_bytes_for_initializers: "1",
    //   },
    // },
  };

  // Debugging if localhost
  // if (location.hostname === "localhost") {
  //   ort.env.debug = true;
  //   ort.env.logLevel = "verbose";
  //   ort.env.trace = true;
  //   options["logSeverityLevel"] = 0;
  //   options["logVerbosityLevel"] = 0;
  // }

  // Create the InferenceSession with the model ArrayBuffer
  const session = await ort.InferenceSession.create(modelArray.buffer, options);
  console.log("Model Output names", session.outputNames);

  return { id, session, genes, classes };
}

// Compute the source indices within sample gene space and destination within
// the model gene space for each gene in the sample gene list
// These are used to populate a sample gene expression tensor into the model
function precomputeInflationIndices(currentModelGenes, sampleGenes) {
  let inflationIndices = [];
  for (let geneIndex = 0; geneIndex < sampleGenes.length; geneIndex++) {
    inflationIndices.push(currentModelGenes.indexOf(sampleGenes[geneIndex]));
  }
  const missingGenesInModel = inflationIndices.filter((x) => x === -1).length;
  console.log(`Missing genes in model: ${missingGenesInModel}`);
  return inflationIndices;
}

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

async function deleteAllDatasets() {
  const request = indexedDB.open("sims-web", 1);
  request.onsuccess = () => {
    const db = request.result;
    if (db.objectStoreNames.contains("datasets")) {
      const tx = db.transaction("datasets", "readwrite");
      const store = tx.objectStore("datasets");
      const getAllRequest = store.getAll();
      getAllRequest.onsuccess = () => {
        const datasets = getAllRequest.result || [];
        datasets.forEach((d) => {
          console.log("Deleting dataset:", d);
          store.delete(d.datasetLabel);
        });
      };
      tx.oncomplete = () => db.close();
    }
  };
  request.onerror = () => {
    console.error("IndexedDB error", request.error);
  };
}

async function predict(event) {
  await deleteAllDatasets();

  self.postMessage({ type: "status", message: "Loading libraries..." });
  const Module = await h5wasm.ready;
  const { FS } = Module;
  console.log("h5wasm loaded");

  try {
    if (!self.model || self.model.id !== event.data.modelID) {
      self.model = await instantiateModel(
        event.data.modelURL,
        event.data.modelID
      );
    }

    // Reset attention accumulators
    self.attentionAccumulators = new Float32Array(
      self.model.classes.length * self.model.genes.length
    );

    self.postMessage({ type: "status", message: "Loading file" });
    if (!FS.analyzePath("/work").exists) {
      FS.mkdir("/work");
    }
    FS.mount(FS.filesystems.WORKERFS, { files: [event.data.h5File] }, "/work");

    const annData = new h5wasm.File(`/work/${event.data.h5File.name}`, "r");
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
    cellNames = cellNames.slice(
      0,
      (event.data.cellRangePercent * cellNames.length) / 100
    );

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

    // Fill the first buffer
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

    // Begin processing batches of cells
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

        // Only push up to maxNumCoords so we limit the memory consumption as these
        // are 32 float vectors per cell
        if (overallCellIndex < self.maxNumCoords) {
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
        type: "progress",
        message: `Predicting ${cellNames.length} out of ${totalNumCells}...`,
        countFinished: nextStart,
        totalToProcess: cellNames.length,
      });

      // Swap buffers
      activeBuffer = nextBuffer;
    }

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
          type: "progress",
          message: `Computing the first ${encodings.length} coordinates...`,
          countFinished: epochNumber,
          totalToProcess: umap.getNEpochs(),
        });
      });
    } catch (error) {
      self.postMessage({ type: "error", error });
      throw error;
    }

    // ========================================================================
    // Calculate top K gene indices per class and overall as well as
    // the overall top k gene indices for all predictions
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
          self.K
        )
      );
      for (let j = 0; j < self.model.genes.length; j++) {
        overallAccumulator[j] +=
          self.attentionAccumulators[i * self.model.genes.length + j];
      }
    }
    let overallTopGenes = topKIndices(overallAccumulator, self.K);

    const cellTypes = labels.map((label) => label[0][0]);

    // Store results in IndexedDB
    const request = indexedDB.open("sims-web", 1);
    request.onsuccess = () => {
      const db = request.result;
      const tx = db.transaction("datasets", "readwrite");
      const store = tx.objectStore("datasets");
      store.put({
        // ! Used by UCSC Cell Browser
        datasetLabel: event.data.h5File.name,
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
      tx.oncomplete = () => db.close();
    };
    request.onerror = () => {
      console.error("IndexedDB error", request.error);
    };

    self.postMessage({
      type: "predictions",
      elapsedTime,
      totalProcessed: cellNames.length,
      totalNumCells,
      genes: self.model.genes,
    });
  } catch (error) {
    // FS.unmount("/work");
    self.postMessage({ type: "error", error: error });
  }
}
