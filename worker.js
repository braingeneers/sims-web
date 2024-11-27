self.importScripts(
    "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js",
    "https://cdn.jsdelivr.net/npm/h5wasm@0.7.8/dist/iife/h5wasm.min.js"
);

function precomputeInflationIndices(currentModelGenes, sampleGenes) {
    let inflationIndices = [];
    for (let geneIndex = 0; geneIndex < sampleGenes.length; geneIndex++) {
        inflationIndices.push(currentModelGenes.indexOf(sampleGenes[geneIndex]));
    }
    return inflationIndices
}

function inflateGenes(inflationIndices, inputTensor, cellIndex, currentModelGenes, sampleGenes, sampleExpression) {
    for (let geneIndex = 0; geneIndex < sampleGenes.length; geneIndex++) {
        inputTensor.data[inflationIndices[geneIndex]] =
            sampleExpression[cellIndex * sampleGenes.length + geneIndex];
    }
}

self.onmessage = async function(event) {
    try {
        const { FS } = await h5wasm.ready;
        console.log("h5wasm loaded");

        // Load the model gene list
        const response = await fetch(`models/${event.data.modelName}.genes`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const currentModelGenes = (await response.text()).split('\n');

        // Load the model
        self.postMessage({ type: 'status', message: 'Loading model' });
        ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
        // ort.env.numThreads = 16;
        // ort.env.proxy = true;
        // ort.env.debug = true;
        // ort.env.logLevel = 'verbose';
        // ort.env.trace = true;
        let options = { executionProviders: ['cpu'] }
        // options['logSeverityLevel'] = 0;
        // options['logVerbosityLevel'] = 0;
        const currentModelSession = await ort.InferenceSession.create(
            `models/${event.data.modelName}.onnx`, options
        );
        console.log('Model loaded');
        console.log('Output names', currentModelSession.outputNames);

        self.postMessage({ type: 'status', message: 'Loading file' });
        FS.mkdir('/work');
        FS.mount(FS.filesystems.WORKERFS, { files: [event.data.h5adFile] }, '/work');

        const  annData= new h5wasm.File(`/work/${event.data.h5adFile.name}`, 'r');
        console.log(annData);

        console.log(`Top level keys: ${annData.keys()}`);
        let cellNames = annData.get('obs').type == 'Dataset' ? 
            annData.get('obs').value.map((e) => e[0]) : annData.get('obs/index').value;
        const totalNumCells = cellNames.length;
        cellNames = cellNames.slice(0, event.data.cellRangePercent * cellNames.length / 100);

        const sampleGenes = annData.get('var').type == 'Dataset' ? 
            annData.get('var').value.map((e) => e[0]) : annData.get('var/index').value;
        const sampleExpression = annData.get('X').value;

        // Depends on the tensor to be zero, and that each cell inflates the same genes
        let inputTensor = new ort.Tensor('float32', new Float32Array(currentModelGenes.length), [1, currentModelGenes.length]);

        const predictions = [];
        const inflationIndices = precomputeInflationIndices(currentModelGenes, sampleGenes);

        const startTime = Date.now(); // Record start time

        for (let cellIndex = 0; cellIndex < cellNames.length; cellIndex++) {
            // const inputTensor = new ort.Tensor('float32', new Float32Array(currentModelGenes.length), [1, currentModelGenes.length]);


            inflateGenes(inflationIndices, inputTensor, cellIndex, currentModelGenes, sampleGenes, sampleExpression);

            const output = await currentModelSession.run({ "input.1": inputTensor });
            // console.log(`${cellNames[cellIndex]} logits: ${output["826"].cpuData}`);

            const argMax = Number(output["argmax"].cpuData[0]);
            const softmax = output["softmax"].cpuData[argMax];
            predictions.push([argMax, softmax]);

            // Post progress update
            const countFinished = cellIndex + 1;
            self.postMessage({ type: 'progress', countFinished, totalToProcess: cellNames.length });
        }

        const endTime = Date.now(); // Record end time
        const elapsedTime = (endTime - startTime) / 60000; // Calculate elapsed time in minutes

        // Post final result
        self.postMessage({ type: 'result', totalNumCells, totalToProcess: cellNames.length, cellNames, predictions, elapsedTime });
    } catch (error) {
        self.postMessage({ type: 'error', error: error.message });
    }
};