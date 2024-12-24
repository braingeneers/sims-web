import React, { useState, useEffect, useRef } from "react";
import {
  Container,
  Typography,
  Button,
  IconButton,
  Select,
  MenuItem,
  InputLabel,
  FormControl,
  Slider,
  LinearProgress,
  Box,
  Link,
  List,
  ListItem,
  ListItemText,
} from "@mui/material";

import { MuiFileInput } from "mui-file-input";
import NewspaperIcon from "@mui/icons-material/Newspaper";
import GitHubIcon from "@mui/icons-material/GitHub";
import DownloadIcon from "@mui/icons-material/Download";

import { PredictionsTable, downloadCSV } from "./PredictionsTable";

import * as d3 from "d3";

import SIMSWorker from "./worker?worker";

function App() {
  const [modelInfoList, setModelInfoList] = useState([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [selectedFile, setSelectedFile] = useState(null);
  const [cellRangePercent, setCellRangePercent] = useState(25);
  const [statusMessage, setStatusMessage] = useState("");
  const [progress, setProgress] = useState(0);
  const [isPredicting, setIsPredicting] = useState(false);
  const [topGenes, setTopGenes] = useState([]);
  const [predictions, setPredictions] = useState(null);
  const [workerInstance, setWorkerInstance] = useState(null);

  const scatterPlotRef = useRef(null);
  const resultsRef = useRef(null);

  // On mount, load models list
  useEffect(() => {
    fetchModels();
    fetchSampleFile();
  }, []);

  // Fill in a sample file so a user can just hit predict to try out
  async function fetchSampleFile() {
    // const sitePath =
    //   window.location.origin +
    //   window.location.pathname.slice(
    //     0,
    //     window.location.pathname.lastIndexOf("/")
    //   );
    // const fileUrl = `${sitePath}/sample.h5ad`;
    const fileUrl = "sample.h5ad";
    const fileName = "sample.h5ad";

    try {
      const response = await fetch(fileUrl);
      const blob = await response.blob();
      const file = new File([blob], fileName, { type: blob.type });
      setSelectedFile(file);
      console.log("File:", file);
    } catch (error) {
      console.error("Error:", error);
    }
  }

  async function fetchModels() {
    try {
      const resp = await fetch("models/models.txt");
      const text = await resp.text();
      const modelIDs = text.split("\n").filter((id) => id.trim() !== "");
      const modelDescriptions = [];
      for (const mid of modelIDs) {
        try {
          const r = await fetch(`models/${mid}.json`);
          if (!r.ok) {
            throw new Error(`Failed to fetch ${mid}.json`);
          }
          const desc = await r.json();
          desc.id = mid;
          modelDescriptions.push(desc);
        } catch (e) {
          console.error(e);
        }
      }
      setModelInfoList(modelDescriptions);
      setSelectedModel("default");
    } catch (err) {
      console.error("Failed fetching model list:", err);
    }
  }

  // Start prediction
  async function handlePredict() {
    if (!selectedModel || !selectedFile) {
      setStatusMessage("Please select a model and file.");
      return;
    }
    if (workerInstance) {
      workerInstance.terminate();
    }
    const newWorker = new SIMSWorker();
    setWorkerInstance(newWorker);
    setIsPredicting(true);
    setStatusMessage("Starting prediction...");
    setProgress(0);

    // Communicate with web worker
    newWorker.onmessage = (evt) => {
      const data = evt.data;
      switch (data.type) {
        case "status":
          setStatusMessage(data.message);
          break;
        case "progress":
          setStatusMessage(data.message);
          setProgress(
            Math.round((data.countFinished / data.totalToProcess) * 100)
          );
          break;
        case "predictions":
          setPredictions(data);
          setStatusMessage(
            `Processed ${data.totalToProcess} of ${
              data.totalNumCells
            } cells in ${data.elapsedTime?.toFixed(2)} minutes`
          );
          // Create scatter plot
          drawScatterPlot(data.coordinates, data.labels);
          // You can also build your HTML table here if you like
          // or store the results in state to display in a React component.
          // After we have results, ask for attention accumulator
          newWorker.postMessage({ type: "getAttentionAccumulator" });
          setIsPredicting(false);
          break;
        case "attentionAccumulator":
          updateTopGenes(data.attentionAccumulator, data.genes);
          break;
        case "error":
          setStatusMessage("Error: " + data.error);
          setIsPredicting(false);
          break;
        default:
          break;
      }
    };

    newWorker.postMessage({
      type: "startPrediction",
      modelID: selectedModel,
      h5File: selectedFile,
      cellRangePercent,
    });
  }

  // Stop prediction
  function handleStop() {
    if (workerInstance) {
      workerInstance.terminate();
      setWorkerInstance(null);
    }
    setStatusMessage("Prediction stopped.");
    setIsPredicting(false);
    setProgress(0);
  }

  // D3 scatter plot
  function drawScatterPlot(coordinates, predictions) {
    if (!scatterPlotRef.current) return;
    // Clear existing
    d3.select(scatterPlotRef.current).selectAll("*").remove();

    const margin = { top: 10, right: 30, bottom: 30, left: 40 };
    const width = 460 - margin.left - margin.right;
    const height = 400 - margin.top - margin.bottom;

    const svg = d3
      .select(scatterPlotRef.current)
      .append("svg")
      .attr("width", width + margin.left + margin.right)
      .attr("height", height + margin.top + margin.bottom)
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    const x = d3
      .scaleLinear()
      .domain(d3.extent(coordinates, (d) => d[0]))
      .range([0, width]);
    const y = d3
      .scaleLinear()
      .domain(d3.extent(coordinates, (d) => d[1]))
      .range([height, 0]);
    svg
      .append("g")
      .attr("transform", `translate(0,${height})`)
      .call(d3.axisBottom(x));
    svg.append("g").call(d3.axisLeft(y));

    const color = d3.scaleOrdinal(d3.schemeCategory10);
    svg
      .selectAll("circle")
      .data(coordinates)
      .enter()
      .append("circle")
      .attr("cx", (d) => x(d[0]))
      .attr("cy", (d) => y(d[1]))
      .attr("r", 3)
      .style("fill", (d, i) => color(predictions[i][0][0]));
  }

  // Parse top genes
  function updateTopGenes(attentionAccumulator, genes) {
    const N = 10;
    const indices = Array.from(attentionAccumulator.keys());
    indices.sort((a, b) => attentionAccumulator[b] - attentionAccumulator[a]);
    setTopGenes(indices.slice(0, N).map((item) => genes[item]));
  }

  // Render
  return (
    <Container maxWidth="lg" sx={{ my: 4 }}>
      <Box display="flex" justifyContent="space-between" alignItems="center">
        <Typography variant="h4">SIMS Web</Typography>
        <Box>
          <Link
            href="https://www.cell.com/cell-genomics/abstract/S2666-979X(24)00165-4"
            target="_blank"
            underline="none"
            sx={{ mr: 2 }}
          >
            <NewspaperIcon />
          </Link>
          <Link
            href="https://github.com/braingeneers/sims-web"
            target="_blank"
            underline="none"
          >
            <GitHubIcon />
          </Link>
        </Box>
      </Box>

      <Box mt={3} mb={3}>
        {/* Model Selection */}
        <FormControl fullWidth margin="normal">
          <InputLabel id="model-label">Select a model</InputLabel>
          <Select
            labelId="model-label"
            label="Select a model"
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
          >
            {modelInfoList.map((m) => (
              <MenuItem key={m.id} value={m.id}>
                {m.title}, {m.submitter}
              </MenuItem>
            ))}
          </Select>
        </FormControl>

        <MuiFileInput
          fullWidth
          placeholder="Select an AnnData/Scanpy (.h5ad) file"
          value={selectedFile}
          onChange={setSelectedFile}
          inputProps={{ accept: ".h5ad" }}
        />

        {/* Slider */}
        <Box mt={2}>
          <Typography>% Of Cells To Process: {cellRangePercent}%</Typography>
          <Slider
            min={0}
            max={100}
            step={1}
            value={cellRangePercent}
            onChange={(e, val) => setCellRangePercent(val)}
            valueLabelDisplay="auto"
          />
        </Box>

        <Box mt={2}>
          <Button
            variant="contained"
            onClick={handlePredict}
            disabled={isPredicting}
          >
            Predict
          </Button>
          <Button
            variant="contained"
            color="error"
            onClick={handleStop}
            sx={{ ml: 2 }}
          >
            Stop
          </Button>
          <IconButton
            onClick={() => downloadCSV(predictions)}
            disabled={predictions === null}
            color="primary"
            style={{ float: "right" }}
          >
            <DownloadIcon />
          </IconButton>
        </Box>
      </Box>

      {/* Status and Progress */}
      <Typography>{statusMessage}</Typography>
      {isPredicting && (
        <Box my={2}>
          <LinearProgress variant="determinate" value={progress} />
          <Typography>{progress}%</Typography>
        </Box>
      )}

      {/* Layout for top genes + scatter plot */}
      <Box display="flex" mt={4}>
        <Box width="25%" mr={4}>
          <Typography variant="h6">Top 10 Genes</Typography>
          <List dense>
            {topGenes.map((gene) => (
              <ListItem key={gene}>
                <ListItemText primary={`${gene}`} />
              </ListItem>
            ))}
          </List>
        </Box>
        <Box ref={scatterPlotRef} width="75%" minHeight={400}></Box>
      </Box>

      {/* Container for table or results */}
      <Box ref={resultsRef} mt={4}></Box>
      <PredictionsTable predictions={predictions} />
    </Container>
  );
}

export default App;
