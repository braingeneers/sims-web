async function getListOfFileNamesExcludingSuffix(path) {
  try {
    const response = await fetch(path);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const text = await response.text();
    const parser = new DOMParser();
    const doc = parser.parseFromString(text, "text/html");
    const fileElements = doc.querySelectorAll("a"); // Assuming file names are in <a> tags
    const fileNames = Array.from(fileElements).map((el) =>
      el.textContent.trim()
    );
    const fileRoots = fileNames.map((name) => name.split(".")[0]);
    const uniqueFileRoots = [...new Set(fileRoots)];
    return uniqueFileRoots.filter((name) => name !== "");
  } catch (error) {
    console.error("Error fetching files:", error);
    return [];
  }
}

async function populateModelSelect(models) {
  const fileSelect = document.getElementById("model_select");
  fileSelect.innerHTML = ""; // Clear existing options
  models.forEach((file) => {
    const option = document.createElement("option");
    option.value = file;
    option.textContent = file;
    if (file === "default") {
      option.selected = true;
    }
    fileSelect.appendChild(option);
  });
}

let worker;

async function predict(worker, modelName, h5File, cellRangePercent) {
  return new Promise((resolve, reject) => {
    worker.postMessage({ modelName, h5File, cellRangePercent });

    worker.onmessage = function (event) {
      const {
        type,
        message,
        countFinished,
        totalToProcess,
        totalNumCells,
        cellNames,
        classes,
        predictions,
        elapsedTime,
        error,
      } = event.data;

      if (type === "status") {
        document.getElementById("message").textContent = message;
      } else if (type === "progress") {
        document.getElementById("message").textContent = message;
        const progress = Math.round((countFinished / totalToProcess) * 100);
        document.getElementById("progress-bar").style.width = `${progress}%`;
        document.getElementById("progress-bar").textContent = `${progress}%`;
      } else if (type === "result") {
        document.getElementById(
          "message"
        ).textContent = `${totalToProcess} of ${totalNumCells} cells in ${elapsedTime.toFixed(
          2
        )} minutes`;
        resolve([cellNames, classes, predictions]);
      } else if (type === "error") {
        reject(error);
      }
    };

    worker.onerror = function (error) {
      reject(error.message);
    };
  });
}

function outputResults(cellNames, predictionClasses, predictions) {
  const resultsContainer = document.getElementById("results");
  resultsContainer.innerHTML = ""; // Clear previous results

  const table = document.createElement("table");
  table.classList.add("table", "table-striped");

  // Create table header
  const thead = document.createElement("thead");
  const headerRow = document.createElement("tr");
  const cellHeader = document.createElement("th");
  cellHeader.textContent = "Cell";
  headerRow.appendChild(cellHeader);

  for (let i = 0; i < 3; i++) {
    const classHeader = document.createElement("th");
    classHeader.textContent = `Class ${i + 1}`;
    headerRow.appendChild(classHeader);

    const confidenceHeader = document.createElement("th");
    confidenceHeader.textContent = `Prob ${i + 1}`;
    headerRow.appendChild(confidenceHeader);
  }

  thead.appendChild(headerRow);
  table.appendChild(thead);

  // Create table body
  const tbody = document.createElement("tbody");

  cellNames.forEach((cellName, cellIndex) => {
    const row = document.createElement("tr");

    const cellNameCell = document.createElement("td");
    cellNameCell.textContent = cellName;
    row.appendChild(cellNameCell);

    for (let i = 0; i < 3; i++) {
      const classCell = document.createElement("td");
      classCell.textContent = predictionClasses[predictions[cellIndex][0][i]];
      row.appendChild(classCell);

      const classSoftmax = document.createElement("td");
      classSoftmax.textContent = predictions[cellIndex][1][i].toFixed(4);
      row.appendChild(classSoftmax);
    }

    tbody.appendChild(row);
  });

  table.appendChild(tbody);
  resultsContainer.appendChild(table);

  document
    .getElementById("download_link")
    .addEventListener("click", function (event) {
      event.preventDefault();
      let csvContent =
        "cell_id,p0_class,p0_prob,p1_class,p1_prob,p2_class,p2_prob\n";

      cellNames.forEach((cellName, cellIndex) => {
        let cellResults = "";
        for (let i = 0; i < 3; i++) {
          cellResults += `,${
            predictionClasses[predictions[cellIndex][0][i]]
          },${predictions[cellIndex][1][i].toFixed(4)}`;
        }
        csvContent += `${cellName}${cellResults}\n`;
      });

      const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
      const downloadLink = document.createElement("a");
      const url = URL.createObjectURL(blob);
      downloadLink.setAttribute("href", url);
      downloadLink.setAttribute("download", "predictions.csv");
      document.body.appendChild(downloadLink);
      downloadLink.click();
      document.body.removeChild(downloadLink);
    });
}

async function main() {
  const response = await fetch("models/models.txt");
  const modelNames = (await response.text()).split("\n");
  populateModelSelect(modelNames);

  // Update the label to show the file name
  document
    .getElementById("file_input")
    .addEventListener("input", function (event) {
      document.getElementById("file_input_label").innerText =
        event.target.files[0].name;
    });

  // Fill in a sample file so a user can just hit predict to try out
  async function urlToFile(url, fileName) {
    const response = await fetch(url);
    const blob = await response.blob();
    const file = new File([blob], fileName, { type: blob.type });
    return file;
  }

  const sitePath =
    window.location.origin +
    window.location.pathname.slice(
      0,
      window.location.pathname.lastIndexOf("/")
    );
  const fileUrl = `${sitePath}/sample.h5ad`;
  const fileName = "sample.h5ad";

  try {
    const file = await urlToFile(fileUrl, fileName);
    console.log("File:", file);

    // You can now use this file as if it was selected from an HTML input element
    // For example, you can set it to an input element
    const fileInput = document.getElementById("file_input");
    const dataTransfer = new DataTransfer();
    dataTransfer.items.add(file);
    fileInput.files = dataTransfer.files;

    // Update the label to show the file name
    document.getElementById("file_input_label").innerText = file.name;
  } catch (error) {
    console.error("Error:", error);
  }

  // Add slider event handler to update displayed cell count and percentage
  document
    .getElementById("cellRange")
    .addEventListener("input", async function (event) {
      const percent = event.target.value;
      document.getElementById("cellRangeValue").textContent = `${percent}%`;
    });

  document.getElementById("stop_btn").addEventListener("click", () => {
    if (worker) {
      worker.terminate();
      worker = null;
      document.getElementById("progress-bar").textContent = "";
      document.getElementById("progress-bar").style.width = "0%";
      document.getElementById("message").textContent = "Prediction stopped";
    }
  });

  document
    .getElementById("predict_btn")
    .addEventListener("click", async (event) => {
      // Clear results at start of prediction
      document.getElementById("results").innerHTML = "";
      document.getElementById("progress-bar").style.width = "0%";
      document.getElementById("progress-bar").textContent = "";
      document.getElementById("message").textContent = "";

      let modelName = document.getElementById("model_select").value;
      const h5File = document.getElementById("file_input").files[0];
      const cellRangePercent = document.getElementById("cellRange").value;

      try {
        if (!worker) {
          worker = new Worker("worker.js");
        }
        const [cellNames, classes, predictions] = await predict(
          worker,
          modelName,
          h5File,
          cellRangePercent
        );
        outputResults(cellNames, classes, predictions);
      } catch (error) {
        console.error("Error:", error);
      }
    });
}
await main();
