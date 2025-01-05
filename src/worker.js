import h5wasm from "h5wasm";
import Prando from "prando";
import * as UMAP from "umap-js";

import * as ort from "onnxruntime-web";

// Includes WebAssembly backend only
// import * as ort from "onnxruntime-web/wasm";

// Includes WebAssembly single-threaded only, with training support
// import * as ort from "onnxruntime-web/training";

// Global variables
self.model = null;
self.attentionAccumulator = null;

const numThreads = navigator.hardwareConcurrency;
const batchSize = navigator.hardwareConcurrency;

// Limit how many UMAP points we calculate which limits memory by limiting the
// encoding vectors we keep around
self.maxNumCoords = 2000;

// Handle messages from the main thread
self.addEventListener("message", async function (event) {
  if (event.data.type === "startPrediction") {
    predict(event);
  } else if (event.data.type === "getAttentionAccumulator") {
    self.postMessage({
      type: "attentionAccumulator",
      attentionAccumulator: self.attentionAccumulator,
      genes: self.model.genes,
    });
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
  // ort.env.wasm.wasmPaths = "/dist/";
  ort.env.wasm.numThreads = numThreads;
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

async function predict(event) {
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

    // Reset attention accumulator
    self.attentionAccumulator = new Float32Array(self.model.genes.length);

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

    // Begin processing batches of cells
    for (
      let batchStart = 0;
      batchStart < cellNames.length;
      batchStart += batchSize
    ) {
      const batchEnd = Math.min(batchStart + batchSize, cellNames.length);
      const currentBatchSize = batchEnd - batchStart;

      // Create a new Float32Array initialized to all zeros for the entire batch
      const inflatedBatchData = new Float32Array(
        currentBatchSize * self.model.genes.length
      );

      // Fill batchData and inflate in one step
      for (let batchSlot = 0; batchSlot < currentBatchSize; batchSlot++) {
        const cellIndex = batchStart + batchSlot;

        let singleCell = null;

        if (isSparse) {
          const [start, end] = indptr.slice([[cellIndex, cellIndex + 2]]);
          const values = data.slice([[start, end]]);
          const valueIndices = indices.slice([[start, end]]);
          singleCell = new Float32Array(sampleGenes.length);
          for (let j = 0; j < valueIndices.length; j++) {
            singleCell[valueIndices[j]] = values[j];
          }
        } else {
          if (data.shape.length === 1) {
            singleCell = data.slice([
              [
                cellIndex * sampleGenes.length,
                (cellIndex + 1) * sampleGenes.length,
              ],
            ]);
          } else if (data.shape.length === 2) {
            singleCell = data.slice([
              [cellIndex, cellIndex + 1],
              [0, sampleGenes.length],
            ]);
          } else {
            throw new Error("Unsupported data shape");
          }
        }

        // Inflate the batchData into its location in the inflatedBatchData
        for (let geneIndex = 0; geneIndex < sampleGenes.length; geneIndex++) {
          const sampleIndex = inflationIndices[geneIndex];
          if (sampleIndex !== -1) {
            inflatedBatchData[
              batchSlot * self.model.genes.length + sampleIndex
            ] = singleCell[geneIndex];
          }
        }
      }

      // Run inference on this inflatedBatch
      const inputTensor = new ort.Tensor("float32", inflatedBatchData, [
        currentBatchSize,
        self.model.genes.length,
      ]);
      const output = await self.model.session.run({ input: inputTensor });

      // Parse and store results for each cell in the batch
      for (let batchSlot = 0; batchSlot < currentBatchSize; batchSlot++) {
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

        // Accumulate attention for each gene
        if (output.attention) {
          const attSize = output.attention.dims[1];
          for (let attIndex = 0; attIndex < attSize; attIndex++) {
            self.attentionAccumulator[attIndex] +=
              output.attention.data[batchSlot * attSize + attIndex];
          }
        }
      }

      // Post progress update
      self.postMessage({
        type: "progress",
        message: `Predicting ${cellNames.length} out of ${totalNumCells}...`,
        countFinished: batchEnd,
        totalToProcess: cellNames.length,
      });
    }

    annData.close();
    FS.unmount("/work");

    // Record end time and calculate elapsed time of prediction only
    const endTime = Date.now(); // Record end time
    const elapsedTime = (endTime - startTime) / 60000; // Calculate elapsed time in minutes

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

    self.postMessage({
      type: "predictions",
      cellNames,
      classes: self.model.classes,
      labels,
      coordinates,
      elapsedTime,
      totalProcessed: cellNames.length,
      totalNumCells,
    });
  } catch (error) {
    // FS.unmount("/work");
    self.postMessage({ type: "error", error: error });
  }
}
