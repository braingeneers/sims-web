async function getListOfFileNamesExcludingSuffix(path) {
    try {
        const response = await fetch(path);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const text = await response.text();
        const parser = new DOMParser();
        const doc = parser.parseFromString(text, 'text/html');
        const fileElements = doc.querySelectorAll('a'); // Assuming file names are in <a> tags
        const fileNames = Array.from(fileElements).map(el => el.textContent.trim());
        const fileRoots = fileNames.map(name => name.split('.').slice(0, -1).join('.'));
        const uniqueFileRoots = [...new Set(fileRoots)];
        return uniqueFileRoots.filter(name => name !== '');
    } catch (error) {
        console.error('Error fetching files:', error);
        return [];
    }
}

async function populateModelSelect(models) {
    const fileSelect = document.getElementById('model_select');
    fileSelect.innerHTML = ''; // Clear existing options
    models.forEach(file => {
        const option = document.createElement('option');
        option.value = file;
        option.textContent = file;
        fileSelect.appendChild(option);
    });
}

async function getGeneListFromServer(modelName) {
    const response = await fetch(`models/${modelName}.genes`);
    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }
    return (await response.text()).split('\n');
}

async function extractAndInflateCellXGene(annData, allGenes) {
    document.getElementById('results').innerHTML = "Fetching gene list...";

    // prepare inputs. a tensor need its corresponding TypedArray as data
    document.getElementById('results').innerHTML = "Inflating genes...";
    const batchSize = 1;
    const X = new ort.Tensor('float32', new Float32Array(batchSize * allGenes.length), [batchSize, allGenes.length]);

    let sampleGenes = annData.get("var/index").value;
    let sampleExpression = annData.get("X").value;

    // Populate the tensor with the first batch of cells
    for (let geneIndex = 0; geneIndex < sampleGenes.length; geneIndex++) {
        let geneIndexInAllGenes = allGenes.indexOf(sampleGenes[geneIndex]);
        for (let cellIndex = 0; cellIndex < batchSize; cellIndex++) {
            X.data[cellIndex * allGenes.length + geneIndexInAllGenes] =
                sampleExpression[cellIndex * sampleGenes.length + geneIndex];
        }
    }
    console.log("Populated X tensor");
    console.log(`X Cell 0 Expression #44: ${X.cpuData[44]}...`);
    return X
}

async function runInference(session, X) {
    try {
        // feed inputs and run
        document.getElementById('results').innerHTML = "Running inference...";
        console.log("Running inference...");
        const results = await session.run({ "input.1": X });
        console.log(`Cell 0 Predictions: ${results["826"].cpuData.slice(0, 8)}`);
        return results;
    } catch (e) {
        console.log(`Failed to inference ONNX model: ${e}.`);
    }
}

let modelNames = await getListOfFileNamesExcludingSuffix("models")
populateModelSelect(modelNames);

document.getElementById("file_input").addEventListener("input", function (event) {
    document.getElementById("file_input_label").innerText = event.target.files[0].name;
});

import h5wasm from "https://cdn.jsdelivr.net/npm/h5wasm@0.7.8/dist/esm/hdf5_hl.js";
const { FS } = await h5wasm.ready;
console.log("h5wasm loaded");

let currentModelName = null;
let currentModelGenes = null;
let currentModelSession = null;

document.getElementById('upload_btn').addEventListener('click', async (event) => {
    // Load the model
    let selectedModelName = document.getElementById('model_select').value;
    if (selectedModelName !== currentModelName) {
        document.getElementById('results').innerHTML = "Loading model...";
        currentModelName = selectedModelName;
        currentModelGenes = await getGeneListFromServer(currentModelName);
        currentModelSession = await ort.InferenceSession.create(`models/${currentModelName}.onnx`);
    }

    const file = document.getElementById('file_input').files[0];
    const reader = new FileReader();
    reader.onload = async (event) => {
        // Read file and write back to browser file system for h5wasm to parse
        document.getElementById('results').innerHTML = "Reading file...";
        const binaryData = event.target.result;
        FS.writeFile("temp.h5ad", new Uint8Array(binaryData));

        let annData = new h5wasm.File("temp.h5ad", "r");
        console.log(`Top level keys: ${annData.keys()}`);
        console.log(`X shape: ${annData.get("X").shape}`);
        console.log(`X genes: ${annData.get("var/index").value.slice(0, 10)}...`);
        console.log(`X cell 0 First 10 expression values: ${annData.get("X").value.slice(0, 10)}...`);

        let X = await extractAndInflateCellXGene(annData, currentModelGenes);
        let results = await runInference(currentModelSession, X);
        let pred = results["826"].cpuData;
        let output = document.getElementById('results');
        output.innerHTML = '<ul class="list-group">';
        for (let i = 0; i < 8; i++) {
            output.innerHTML += `<li class="list-group-item">${pred[i]}</li>`
        }
        output.innerHTML += '</ul>'

    };
    if (file) {
        reader.readAsArrayBuffer(file);
    }
});