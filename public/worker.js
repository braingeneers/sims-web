self.importScripts(
  "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js",
  "https://cdn.jsdelivr.net/npm/h5wasm@0.7.8/dist/iife/h5wasm.min.js",
  "https://cdn.jsdelivr.net/npm/umap-js@1.4.0/lib/umap-js.min.js"
);

// Global variables
self.model = null;
self.attentionAccumulator = null;

self.addEventListener("message", async function (event) {
  const { type } = event.data;

  if (type === "startPrediction") {
    predict(event);
  } else if (type === "getAttentionAccumulator") {
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
async function instantiateModel(id) {
  self.postMessage({ type: "status", message: "Downloading model..." });

  // Load the model gene list
  let response = await fetch(`models/${id}.genes`);
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  const genes = (await response.text()).split("\n");

  // Load the model classes
  response = await fetch(`models/${id}.classes`);
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  const classes = (await response.text()).split("\n");

  const modelUrl = `models/${id}.onnx`;
  response = await fetch(modelUrl);

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  const contentLength = response.headers.get("Content-Length");
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
    const progress = Math.round((loadedBytes / totalBytes) * 100);

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
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";
  let options = { executionProviders: ["cpu"] };

  if (location.hostname === "localhost") {
    ort.env.debug = true;
    // ort.env.logLevel = "verbose";
    // ort.env.trace = true;
    // options["logSeverityLevel"] = 0;
    // options["logVerbosityLevel"] = 0;
  }

  // Create the InferenceSession with the model ArrayBuffer
  const session = await ort.InferenceSession.create(modelArray.buffer, options);
  console.log("Model Output names", session.outputNames);

  return { id, session, genes, classes };
}

function precomputeInflationIndices(currentModelGenes, sampleGenes) {
  let inflationIndices = [];
  for (let geneIndex = 0; geneIndex < sampleGenes.length; geneIndex++) {
    inflationIndices.push(currentModelGenes.indexOf(sampleGenes[geneIndex]));
  }
  const missingGenesInModel = inflationIndices.filter((x) => x === -1).length;
  console.log(`Missing genes in model: ${missingGenesInModel}`);
  return inflationIndices;
}

function inflateGenes(
  inflationIndices,
  inputTensor,
  cellIndex,
  sampleGenes,
  sampleExpression
) {
  // Slicing is done through libhdf5 javascript - should be very efficient and
  // only read the necessary data thereby enabling unlimited size datasets
  let sampleExpressionSlice = null;
  if (sampleExpression.shape.length === 1) {
    sampleExpressionSlice = sampleExpression.slice([
      [cellIndex * sampleGenes.length, (cellIndex + 1) * sampleGenes.length],
    ]);
  } else if (sampleExpression.shape.length === 2) {
    sampleExpressionSlice = sampleExpression.slice([
      [cellIndex, cellIndex + 1],
      [0, sampleGenes.length],
    ]);
  } else {
    throw new Error("Unsupported expression matrix shape");
  }

  for (let geneIndex = 0; geneIndex < sampleGenes.length; geneIndex++) {
    const sampleIndex = inflationIndices[geneIndex];
    if (sampleIndex !== -1) {
      inputTensor.data[sampleIndex] = sampleExpressionSlice[geneIndex];
    }
  }
}

async function predict(event) {
  self.postMessage({ type: "status", message: "Loading libraries..." });
  const { FS } = await h5wasm.ready;
  console.log("h5wasm loaded");

  try {
    if (!self.model || self.model.id !== event.data.modelID) {
      self.model = await instantiateModel(event.data.modelID);
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
    cellNames = cellNames.slice(
      0,
      (event.data.cellRangePercent * cellNames.length) / 100
    );

    let sampleExpression = null;
    if (annData.get("X").type == "Dataset") {
      sampleExpression = annData.get("X");
    } else if (annData.get("X").type == "Group") {
      sampleExpression = annData.get("X/data");
    }

    // Depends on the tensor to be zero, and that each cell inflates the same genes
    let inputTensor = new ort.Tensor(
      "float32",
      new Float32Array(model.genes.length),
      [1, model.genes.length]
    );

    const predictions = [];
    const encodings = [];
    const inflationIndices = precomputeInflationIndices(
      self.model.genes,
      sampleGenes
    );

    const startTime = Date.now(); // Record start time

    // Begin processing cells
    for (let cellIndex = 0; cellIndex < cellNames.length; cellIndex++) {
      inflateGenes(
        inflationIndices,
        inputTensor,
        cellIndex,
        sampleGenes,
        sampleExpression
      );

      let output = await self.model.session.run({ input: inputTensor });

      predictions.push([output.topk_indices.cpuData, output.probs.cpuData]);

      encodings.push(output.encoding.cpuData);

      for (let i = 0; i < self.attentionAccumulator.length; i++) {
        self.attentionAccumulator[i] += output.attention.cpuData[i];
      }

      // Post progress update
      const countFinished = cellIndex + 1;
      self.postMessage({
        type: "progress",
        message: "Predicting...",
        countFinished,
        totalToProcess: cellNames.length,
      });
    }

    const umap = new UMAP.UMAP({
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
          message: "Computing coordinates...",
          countFinished: epochNumber,
          totalToProcess: umap.getNEpochs(),
        });
      });
    } catch (error) {
      self.postMessage({ type: "error", error });
      throw error;
    }

    annData.close();
    FS.unmount("/work");

    const endTime = Date.now(); // Record end time
    const elapsedTime = (endTime - startTime) / 60000; // Calculate elapsed time in minutes
    // Post final result
    self.postMessage({
      type: "result",
      cellNames,
      classes: self.model.classes,
      predictions,
      coordinates,
      elapsedTime,
      totalToProcess: cellNames.length,
      totalNumCells,
    });
  } catch (error) {
    FS.unmount("/work");
    self.postMessage({ type: "error", error: error });
  }
}
