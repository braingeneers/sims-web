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

async function extractAndInflateCellXGeneAndRunInference(modelName, h5adFile) {
    return new Promise((resolve, reject) => {
        const worker = new Worker('worker.js');

        worker.postMessage({ modelName, h5adFile});

        worker.onmessage = function(event) {
            const { type, cellIndex, totalCells, combinedMatrix, error } = event.data;

            if (type === 'progress') {
                const progress = Math.round((event.data.countFinished / totalCells) * 100);
                document.getElementById('progress-bar').style.width = `${progress}%`;
                document.getElementById('progress-bar').textContent = `${progress}%`;
            } else if (type === 'result') {
                resolve(combinedMatrix);
            } else if (type === 'error') {
                reject(error);
            }
        };

        worker.onerror = function(error) {
            reject(error.message);
        };
    });
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

    let selectedModelName = document.getElementById('model_select').value;

    const h5AdFile = document.getElementById('file_input').files[0];

    try {
        const combinedMatrix = await extractAndInflateCellXGeneAndRunInference(selectedModelName, h5AdFile);
        console.log('Combined Matrix:', combinedMatrix);
        let output = document.getElementById('results');
        output.innerHTML = "";
        output.innerHTML = '<div>Cell 0 Raw Predictions</div>';
        output.innerHTML += '<ul class="list-group">';
        for (let i = 0; i < 8; i++) {
            output.innerHTML += `<li class="list-group-item">${combinedMatrix[0][i]}</li>`
        }
        output.innerHTML += '</ul>'
    } catch (error) {
        console.error('Error:', error);
    }

});