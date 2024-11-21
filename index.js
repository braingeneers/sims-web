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
            const { type, countFinished, totalCells, cellNames, combinedMatrix, elapsedTime, error } = event.data;

            if (type === 'progress') {
                const progress = Math.round((event.data.countFinished / totalCells) * 100);
                document.getElementById('progress-bar').style.width = `${progress}%`;
                document.getElementById('progress-bar').textContent = `${progress}%`;
            } else if (type === 'result') {
                document.getElementById('elapsed-time').textContent = `Elapsed Time: ${elapsedTime.toFixed(2)} minutes`;
                resolve([cellNames, combinedMatrix]);
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


// DEBUGING
// If localhost then fill in a remote file so we can just hit enter vs. selecting each reload
if (location.host === "localhost:3000") {
    async function urlToFile(url, fileName) {
        const response = await fetch(url);
        const blob = await response.blob();
        const file = new File([blob], fileName, { type: blob.type });
        return file;
    }

    const fileUrl = 'http://localhost:3000/data/pbmc3k.h5ad'; // Replace with actual file URL
    const fileName = 'pbmc3k.h5ad'; // Replace with desired file name

    try {
        const file = await urlToFile(fileUrl, fileName);
        console.log('File:', file);

        // You can now use this file as if it was selected from an HTML input element
        // For example, you can set it to an input element
        const fileInput = document.getElementById('file_input');
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        fileInput.files = dataTransfer.files;

        // Update the label to show the file name
        document.getElementById('file_input_label').innerText = file.name;
    } catch (error) {
        console.error('Error:', error);
    }
}

function outputResults(cellNames, combinedMatrix, predictionClasses) {
    const resultsContainer = document.getElementById('results');
    resultsContainer.innerHTML = ''; // Clear previous results

    const table = document.createElement('table');
    table.classList.add('table', 'table-striped');

    // Create table header
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    const cellHeader = document.createElement('th');
    cellHeader.textContent = 'Cell';
    headerRow.appendChild(cellHeader);

    const classHeader = document.createElement('th');
    classHeader.textContent = 'Class';
    headerRow.appendChild(classHeader);

    for (let i = 0; i < 8; i++) {
        const outputHeader = document.createElement('th');
        outputHeader.textContent = `Output ${i + 1}`;
        headerRow.appendChild(outputHeader);
    }

    thead.appendChild(headerRow);
    table.appendChild(thead);

    // Create table body
    const tbody = document.createElement('tbody');

    cellNames.forEach((cellName, cellIndex) => {
        const row = document.createElement('tr');
        const cellNameCell = document.createElement('td');
        cellNameCell.textContent = cellName;
        row.appendChild(cellNameCell);

        const outputValues = combinedMatrix[cellIndex];
        const maxIndex = indexOfMax(outputValues);
        const classLabel = predictionClasses[maxIndex];
        const classCell = document.createElement('td');
        classCell.textContent = classLabel;
        row.appendChild(classCell);

        outputValues.forEach((value, index) => {
            const outputCell = document.createElement('td');
            outputCell.textContent = value.toFixed(4); // Format to 4 decimal places
            row.appendChild(outputCell);
        });

        tbody.appendChild(row);
    });

    table.appendChild(tbody);
    resultsContainer.appendChild(table);
}

function indexOfMax(arr) {
    if (arr.length === 0) {
        return -1;
    }

    let max = arr[0];
    let maxIndex = 0;

    for (let i = 1; i < arr.length; i++) {
        if (arr[i] > max) {
            maxIndex = i;
            max = arr[i];
        }
    }

    return maxIndex;
}

document.getElementById('predict_btn').addEventListener('click', async (event) => {

    let selectedModelName = document.getElementById('model_select').value;

    const h5AdFile = document.getElementById('file_input').files[0];

    // Load the model classes
    const response = await fetch(`models/${selectedModelName}.classes`);
    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }
    const selectedModelClasses = (await response.text()).split('\n');

    try {
        const [cellNames, combinedMatrix] = await extractAndInflateCellXGeneAndRunInference(selectedModelName, h5AdFile);
        console.log('Combined Matrix:', combinedMatrix);
        outputResults(cellNames, combinedMatrix, selectedModelClasses);
    } catch (error) {
        console.error('Error:', error);
    }

});