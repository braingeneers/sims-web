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
        if (file === 'default') {
            option.selected = true;
        }
        fileSelect.appendChild(option);
    });
}

async function predict(modelName, h5adFile, cellRangePercent) {
    return new Promise((resolve, reject) => {
        const worker = new Worker('worker.js');

        worker.postMessage({ modelName, h5adFile, cellRangePercent});

        worker.onmessage = function(event) {
            const { type, message, countFinished, totalToProcess, totalNumCells, cellNames, predictions, elapsedTime, error } = event.data;

            if (type === 'status') {
                document.getElementById('progress-bar').textContent = message;
            } else if (type === 'progress') {
                const progress = Math.round((countFinished / totalToProcess) * 100);
                document.getElementById('progress-bar').style.width = `${progress}%`;
                document.getElementById('progress-bar').textContent = `${progress}%`;
            } else if (type === 'result') {
                document.getElementById('elapsed-time').textContent = `${totalToProcess} of ${totalNumCells} cells in ${elapsedTime.toFixed(2)} minutes`;
                resolve([cellNames, predictions]);
            } else if (type === 'error') {
                reject(error);
            }
        };

        worker.onerror = function(error) {
            reject(error.message);
        };
    });
}

function outputResults(cellNames, predictions, predictionClasses) {
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

    const confidenceHeader = document.createElement('th');
    confidenceHeader.textContent = 'Confidence';
    headerRow.appendChild(confidenceHeader);

    thead.appendChild(headerRow);
    table.appendChild(thead);

    // Create table body
    const tbody = document.createElement('tbody');

    cellNames.forEach((cellName, cellIndex) => {
        const row = document.createElement('tr');

        const cellNameCell = document.createElement('td');
        cellNameCell.textContent = cellName;
        row.appendChild(cellNameCell);

        const classCell = document.createElement('td');
        classCell.textContent = predictionClasses[predictions[cellIndex][0]];
        row.appendChild(classCell);

        const classSoftmax = document.createElement('td');
        classSoftmax.textContent = predictions[cellIndex][1].toFixed(4);
        row.appendChild(classSoftmax);

        tbody.appendChild(row);
    });

    table.appendChild(tbody);
    resultsContainer.appendChild(table);
}

async function main() {
    let modelNames = await getListOfFileNamesExcludingSuffix("models")
    populateModelSelect(modelNames);

    // Update the label to show the file name
    document.getElementById("file_input").addEventListener("input", function (event) {
        document.getElementById("file_input_label").innerText = event.target.files[0].name;
    });

    // DEBUGGING
    // If localhost then fill in a remote file so we can just hit enter vs. selecting each reload
    // and set the percentage to 1% for quick testing
    if (location.host === "localhost:3000") {
        async function urlToFile(url, fileName) {
            const response = await fetch(url);
            const blob = await response.blob();
            const file = new File([blob], fileName, { type: blob.type });
            return file;
        }

        const fileUrl = 'http://localhost:3000/data/default.h5ad'; // Replace with actual file URL
        const fileName = 'default.h5ad'; // Replace with desired file name

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

    // Add slider event handler to update displayed cell count and percentage
    document.getElementById('cellRange').addEventListener('input', async function(event) {
        const percent = event.target.value;
        document.getElementById('cellRangeValue').textContent = `${percent}%`; 
    });

    document.getElementById('predict_btn').addEventListener('click', async (event) => {
        // Clear results at start of prediction
        document.getElementById('results').innerHTML = '';
        document.getElementById('progress-bar').style.width = '100%';
        document.getElementById('progress-bar').textContent = '';
        document.getElementById('elapsed-time').textContent = '';

        let selectedModelName = document.getElementById('model_select').value;
        const h5AdFile = document.getElementById('file_input').files[0];
        const cellRangePercent = document.getElementById('cellRange').value;

        // Load the model classes
        const response = await fetch(`models/${selectedModelName}.classes`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const selectedModelClasses = (await response.text()).split('\n');

        try {
            const [cellNames, predictions] = await predict(selectedModelName, h5AdFile, cellRangePercent);
            outputResults(cellNames, predictions, selectedModelClasses);
        } catch (error) {
            console.error('Error:', error);
        }

    });
}
await main();