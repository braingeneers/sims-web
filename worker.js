self.importScripts(
  "https://cdnjs.cloudflare.com/ajax/libs/onnxruntime-web/1.20.1/ort.min.js",
  "https://cdn.jsdelivr.net/npm/h5wasm@0.7.8/dist/iife/h5wasm.min.js"
);

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
  currentModelGenes,
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

self.onmessage = async function (event) {
  try {
    self.postMessage({ type: "status", message: "Loading libraries..." });
    const { FS } = await h5wasm.ready;
    console.log("h5wasm loaded");

    // Load the model gene list
    const response = await fetch(`models/${event.data.modelName}.genes`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const currentModelGenes = (await response.text()).split("\n");

    // Load the model
    self.postMessage({ type: "status", message: "Loading model..." });
    ort.env.wasm.wasmPaths =
      "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";
    // ort.env.numThreads = 16;
    // ort.env.proxy = true;
    let options = { executionProviders: ["cpu"] };
    if (location.hostname === "localhost") {
      ort.env.debug = true;
      ort.env.logLevel = "verbose";
      ort.env.trace = true;
      options["logSeverityLevel"] = 0;
      options["logVerbosityLevel"] = 0;
    }
    const currentModelSession = await ort.InferenceSession.create(
      `models/${event.data.modelName}.onnx`,
      options
    );
    console.log("Model Output names", currentModelSession.outputNames);

    self.postMessage({ type: "status", message: "Loading file" });
    FS.mkdir("/work");
    FS.mount(FS.filesystems.WORKERFS, { files: [event.data.h5File] }, "/work");

    const annData = new h5wasm.File(`/work/${event.data.h5File.name}`, "r");
    console.log(annData);

    console.log(`Top level keys: ${annData.keys()}`);

    let cellNames = [];
    let sampleGenes = [];
    if (annData.get("obs").type == "Dataset") {
      cellNames = annData.get("obs").value.map((e) => e[0]);
      sampleGenes = annData.get("var").value.map((e) => e[0]);
    } else if (annData.get("obs").type == "Group") {
      if (annData.get("obs").keys().includes("index")) {
        cellNames = annData.get("obs/index").value;
        sampleGenes = annData.get("var/index").value;
      } else if (annData.get("obs").keys().includes("_index")) {
        cellNames = annData.get("obs/_index").value;
        sampleGenes = annData.get("var/_index").value;
      } else {
        throw new Error("Could not find cell names");
      }
    } else {
      throw new Error("Could not find cell names");
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
      new Float32Array(currentModelGenes.length),
      [1, currentModelGenes.length]
    );

    const predictions = [];
    const inflationIndices = precomputeInflationIndices(
      currentModelGenes,
      sampleGenes
    );

    const startTime = Date.now(); // Record start time

    for (let cellIndex = 0; cellIndex < cellNames.length; cellIndex++) {
      inflateGenes(
        inflationIndices,
        inputTensor,
        cellIndex,
        currentModelGenes,
        sampleGenes,
        sampleExpression
      );

      let output = await currentModelSession.run({ input: inputTensor });
      predictions.push([output.topk_indices.cpuData, output.probs.cpuData]);

      // Post progress update
      const countFinished = cellIndex + 1;
      self.postMessage({
        type: "progress",
        countFinished,
        totalToProcess: cellNames.length,
      });
    }

    const endTime = Date.now(); // Record end time
    const elapsedTime = (endTime - startTime) / 60000; // Calculate elapsed time in minutes

    // Post final result
    self.postMessage({
      type: "result",
      totalNumCells,
      totalToProcess: cellNames.length,
      cellNames,
      predictions,
      elapsedTime,
    });
  } catch (error) {
    self.postMessage({ type: "error", error: error.message });
  }
};
