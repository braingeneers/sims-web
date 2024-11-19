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

async function extractAndInflateCellXGeneAndRunInference(annData, allGenes, session) {
    try {
        const sampleGenes = annData.get('var/index').value;
        const cellNames = annData.get('obs/index').value;
        const sampleExpression = annData.get('X').value;

        const combinedMatrix = [];

        // for (let cellIndex = 0; cellIndex < cellNames.length; cellIndex++) {
        for (let cellIndex = 0; cellIndex < 100; cellIndex++) {

            document.getElementById('results').innerHTML = `Processing cell ${cellIndex}...`;
            const cellData = sampleExpression.slice(cellIndex * sampleGenes.length, (cellIndex + 1) * sampleGenes.length);
            const inputTensor = new ort.Tensor('float32', new Float32Array(allGenes.length), [1, allGenes.length]);

            // Populate the tensor with the first batch of cells
            for (let geneIndex = 0; geneIndex < sampleGenes.length; geneIndex++) {
                let geneIndexInAllGenes = allGenes.indexOf(sampleGenes[geneIndex]);
                inputTensor.data[cellIndex * allGenes.length + geneIndexInAllGenes] =
                    sampleExpression[cellIndex * sampleGenes.length + geneIndex];
            }

            const results = await session.run({ "input.1": inputTensor });
            const output = results["826"].cpuData

            combinedMatrix.push(output);
        }

        return combinedMatrix;
    } catch (error) {
        console.error('Error extracting, inflating, and running inference on cell x gene data:', error);
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
    let output = document.getElementById('results');
    output.innerHTML = "";

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

        const results = await extractAndInflateCellXGeneAndRunInference(annData, currentModelGenes,currentModelSession);

        output.innerHTML = '<div>Cell 0 Raw Predictions</div>';
        output.innerHTML += '<ul class="list-group">';
        for (let i = 0; i < 8; i++) {
            output.innerHTML += `<li class="list-group-item">${results[0][i]}</li>`
        }
        output.innerHTML += '</ul>'

    };
    if (file) {
        reader.readAsArrayBuffer(file);
    }
});