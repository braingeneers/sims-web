import { useState, useEffect, useRef } from "react";
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

import { PredictionsTable } from "./PredictionsTable";
import { PredictionsPlot } from "./PredictionsPlot";

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

  const resultsRef = useRef(null);

  function downloadCSV(predictions) {
    if (!predictions) {
      return;
    }
    let csvContent =
      "cell_id,pred_0,pred_1,pred_2,prob_0,prob_1,prob_2,umap_0,umap_1\n";

    predictions.cellNames.forEach((cellName, cellIndex) => {
      let cellResults = "";
      for (let i = 0; i < 3; i++) {
        cellResults += `,${
          predictions.classes[predictions.labels[cellIndex][0][i]]
        }`;
      }
      for (let i = 0; i < 3; i++) {
        cellResults += `,${predictions.labels[cellIndex][1][i].toFixed(4)}`;
      }
      if (cellIndex < predictions.coordinates.length) {
        cellResults += `,${predictions.coordinates[cellIndex][0].toFixed(
          4
        )},${predictions.coordinates[cellIndex][1].toFixed(4)}`;
      } else {
        // If there are no UMAP coordinates, just add a comma
        cellResults += ",,";
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
  }

  const sitePath =
    window.location.origin +
    window.location.pathname.slice(
      0,
      window.location.pathname.lastIndexOf("/")
    );

  // On mount, load models list
  useEffect(() => {
    fetchModels();
    fetchSampleFile();
  }, []);

  // Fill in a sample file so a user can just hit predict to try out
  async function fetchSampleFile() {
    try {
      // const sampleFileName = "sample.h5ad";
      const sampleFileName = "sample-sparse.h5ad";
      const response = await fetch(sampleFileName);
      const blob = await response.blob();
      const file = new File([blob], sampleFileName, { type: blob.type });
      setSelectedFile(file);
      console.log("Sample File:", file);
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
    // Clear existing output
    setTopGenes([]);
    setPredictions(null);
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
            `Processed ${data.totalProcessed} of ${
              data.totalNumCells
            } cells in ${data.elapsedTime?.toFixed(2)} minutes`
          );
          newWorker.postMessage({ type: "getAttentionAccumulator" });
          setIsPredicting(false);
          break;
        case "attentionAccumulator":
          updateTopGenes(data.attentionAccumulator, data.genes);
          break;
        case "error":
          setStatusMessage(data.error.toString());
          setIsPredicting(false);
          break;
        default:
          break;
      }
    };

    newWorker.postMessage({
      type: "startPrediction",
      modelsURL: `${sitePath}/models`,
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
          <Typography variant="h6">
            {topGenes.length ? "Top 10 Genes" : ""}
          </Typography>
          <List dense>
            {topGenes.map((gene) => (
              <ListItem key={gene}>
                <ListItemText primary={`${gene}`} />
              </ListItem>
            ))}
          </List>
        </Box>
        <PredictionsPlot width={450} height={450} predictions={predictions} />
      </Box>

      {/* Container for table or results */}
      <Box ref={resultsRef} mt={4}></Box>
      <PredictionsTable predictions={predictions} />
    </Container>
  );
}

export default App;
