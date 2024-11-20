self.importScripts("https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js", "dist/h5wasm/iife/h5wasm.js");


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
        ort.env.wasm.wasmPaths = {
            wasm: `${location.origin}/node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm`,
            mjs: `${location.origin}/node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs`
        };
        const currentModelSession = await ort.InferenceSession.create(`models/${event.data.modelName}.onnx`);

        // Load the h5file
        FS.mkdir('/work');
        FS.mount(FS.filesystems.WORKERFS, { files: [event.data.h5adFile] }, '/work');

        const  annData= new h5wasm.File(`/work/${event.data.h5adFile.name}`, 'r');
        console.log(annData);

        console.log(`Top level keys: ${annData.keys()}`);
        console.log(`X shape: ${annData.get("X").shape}`);
        console.log(`X genes: ${annData.get("var/index").value.slice(0, 10)}...`);
        console.log(`X cell 0 First 10 expression values: ${annData.get("X").value.slice(0, 10)}...`);
        const sampleGenes = annData.get('var/index').value;
        // const cellNames = annData.get('obs/index').value;
        const cellNames = annData.get('obs/index').value.slice(0, 10);
        const sampleExpression = annData.get('X').value;

        // Depending on the tensor to be zero, and that each cell inflates the same genes
        const inputTensor = new ort.Tensor('float32', new Float32Array(currentModelGenes.length), [1, currentModelGenes.length]);

        const combinedMatrix = [];

        for (let cellIndex = 0; cellIndex < cellNames.length; cellIndex++) {

            // Populate the tensor with the first batch of cells
            for (let geneIndex = 0; geneIndex < sampleGenes.length; geneIndex++) {
                let geneIndexInAllGenes = currentModelGenes.indexOf(sampleGenes[geneIndex]);
                inputTensor.data[cellIndex * currentModelGenes.length + geneIndexInAllGenes] =
                    sampleExpression[cellIndex * sampleGenes.length + geneIndex];
            }

            const results = await currentModelSession.run({ "input.1": inputTensor });
            const output = results["826"].cpuData

            combinedMatrix.push(output);

            // Post progress update
            const countFinished = cellIndex + 1;
            self.postMessage({ type: 'progress', countFinished, totalCells: cellNames.length });
        }

        // Post final result
        self.postMessage({ type: 'result', combinedMatrix });
    } catch (error) {
        self.postMessage({ type: 'error', error: error.message });
    }
};