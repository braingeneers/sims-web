async function fetchModelDescriptions(modelIDs) {
  const modelDescriptions = [];

  for (const modelID of modelIDs) {
    try {
      const response = await fetch(`models/${modelID}.json`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const description = await response.json();
      description["id"] = modelID;
      modelDescriptions.push(description);
    } catch (error) {
      console.error(`Failed to fetch description for ${modelID}:`, error);
    }
  }

  return modelDescriptions;
}

async function populateModelSelect(modelDescriptions) {
  const fileSelect = document.getElementById("model_select");
  fileSelect.innerHTML = ""; // Clear existing options

  modelDescriptions.forEach((description) => {
    const option = document.createElement("option");
    option.value = description.id;
    option.textContent = `${description.title}, ${description.submitter}`;
    fileSelect.appendChild(option);
  });
}

let worker = null;

async function predict(worker, modelID, h5File, cellRangePercent) {
  return new Promise((resolve, reject) => {
    worker.postMessage({ modelID, h5File, cellRangePercent });

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
        encodings,
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
        resolve([cellNames, classes, predictions, encodings]);
      } else if (type === "error") {
        reject(error);
      }
    };

    worker.onerror = function (error) {
      reject(error.message);
    };
  });
}

function createScatterPlot(encodings) {
  // Set the dimensions and margins of the graph
  const margin = { top: 10, right: 30, bottom: 30, left: 40 },
    width = 460 - margin.left - margin.right,
    height = 400 - margin.top - margin.bottom;

  // Append the svg object to the body of the page
  const svg = d3
    .select("#scatter-plot")
    .append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
    .append("g")
    .attr("transform", `translate(${margin.left},${margin.top})`);

  // Add X axis
  const x = d3
    .scaleLinear()
    .domain([d3.min(encodings, (d) => d[0]), d3.max(encodings, (d) => d[0])])
    .range([0, width]);
  svg
    .append("g")
    .attr("transform", `translate(0,${height})`)
    .call(d3.axisBottom(x));

  // Add Y axis
  const y = d3
    .scaleLinear()
    .domain([d3.min(encodings, (d) => d[1]), d3.max(encodings, (d) => d[1])])
    .range([height, 0]);
  svg.append("g").call(d3.axisLeft(y));

  // Add dots
  svg
    .append("g")
    .selectAll("dot")
    .data(encodings)
    .enter()
    .append("circle")
    .attr("cx", (d) => x(d[0]))
    .attr("cy", (d) => y(d[1]))
    .attr("r", 3)
    .style("fill", "#69b3a2");
}

function outputResults(cellNames, predictionClasses, predictions) {
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
  document.getElementById("results").appendChild(table);

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
  const modelIDs = (await response.text()).split("\n");
  const modelDescriptions = await fetchModelDescriptions(modelIDs);
  populateModelSelect(modelDescriptions);

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
      document.getElementById("scatter-plot").innerHTML = "";
      document.getElementById("results").innerHTML = "";
      document.getElementById("progress-bar").style.width = "0%";
      document.getElementById("progress-bar").textContent = "";
      document.getElementById("message").textContent = "";

      let modelID = document.getElementById("model_select").value;
      const h5File = document.getElementById("file_input").files[0];
      const cellRangePercent = document.getElementById("cellRange").value;

      try {
        if (!worker) {
          worker = new Worker("worker.js");
        }
        const [cellNames, classes, predictions, encodings] = await predict(
          worker,
          modelID,
          h5File,
          cellRangePercent
        );
        createScatterPlot(encodings);
        outputResults(cellNames, classes, predictions);
      } catch (error) {
        console.error("Error:", error);
      }
    });
}
await main();
