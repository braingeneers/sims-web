import { useState, useEffect, useRef } from "react";
import {
  Alert,
  Container,
  Typography,
  Button,
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
  IconButton,
} from "@mui/material";

import { MuiFileInput } from "mui-file-input";
import NewspaperIcon from "@mui/icons-material/Newspaper";
import DeleteIcon from "@mui/icons-material/Delete";

import GitHubIcon from "@mui/icons-material/GitHub";

import { PredictionsTable } from "./PredictionsTable";
import { PredictionsPlot } from "./PredictionsPlot";
import { PredictionsSankey } from "./PredictionsSankey";

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
  const [dbDatasets, setDbDatasets] = useState([]);

  const resultsRef = useRef(null);

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
    loadDatasets();
  }, []);

  // Load datasets from IndexedDB
  async function loadDatasets() {
    const request = indexedDB.open("sims-web", 1);
    request.onupgradeneeded = (event) => {
      console.log("Creating results database");
      const db = event.target.result;
      db.createObjectStore("datasets", { keyPath: "datasetLabel" });
    };
    request.onsuccess = () => {
      const db = request.result;
      if (db.objectStoreNames.contains("datasets")) {
        const tx = db.transaction("datasets", "readonly");
        const store = tx.objectStore("datasets");
        const getAllRequest = store.getAll();
        getAllRequest.onsuccess = () => {
          const result = getAllRequest.result || [];
          setDbDatasets(result);
          db.close();
        };
      } else {
        console.log("No existing results found");
      }
    };
  }

  // Delete a dataset
  function handleDeleteDataset(label) {
    console.log("Deleting results", label);
    const request = indexedDB.open("sims-web", 1);
    request.onsuccess = () => {
      const db = request.result;
      const tx = db.transaction("datasets", "readwrite");
      const store = tx.objectStore("datasets");
      store.delete(label);
      tx.oncomplete = () => {
        db.close();
        loadDatasets();
      };
    };
  }

  // Fill in a sample file so a user can just hit predict to try out
  async function fetchSampleFile() {
    try {
      const sampleFileName = "sample.h5ad";
      // const sampleFileName = "sample-sparse.h5ad";
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

  useEffect(() => {
    const worker = new SIMSWorker();
    setWorkerInstance(worker);

    worker.onmessage = (evt) => {
      switch (evt.data.type) {
        case "status":
          setStatusMessage(evt.data.message);
          break;
        case "progress":
          setStatusMessage(evt.data.message);
          setProgress(
            Math.round((evt.data.countFinished / evt.data.totalToProcess) * 100)
          );
          break;
        case "predictions":
          setPredictions(evt.data);
          setStatusMessage(
            `Processed ${evt.data.totalProcessed} of ${
              evt.data.totalNumCells
            } cells in ${evt.data.elapsedTime?.toFixed(2)} minutes`
          );
          setTopGenes(
            evt.data.overallTopGenes.map((item) => evt.data.genes[item])
          );
          setIsPredicting(false);
          loadDatasets(); // Refresh the list of stored datasets
          break;
        case "error":
          setStatusMessage(evt.data.error.toString());
          setIsPredicting(false);
          break;
        default:
          break;
      }

      return () => {
        // Cleanup
        worker.terminate();
      };
    };
  }, []);

  // Start prediction
  async function handlePredict() {
    if (!workerInstance) return;
    // Clear existing output
    setProgress(0);
    setTopGenes([]);
    setPredictions(null);
    if (!selectedModel || !selectedFile) {
      setStatusMessage("Please select a model and file.");
      return;
    }
    setIsPredicting(true);
    setStatusMessage("Starting prediction...");
    workerInstance.postMessage({
      type: "startPrediction",
      modelURL: `${sitePath}/models`,
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
        </Box>
      </Box>

      <Typography variant="h5" sx={{ mt: 4 }}>
        Results
      </Typography>
      <List dense>
        {dbDatasets.map((ds) => (
          <ListItem key={ds.datasetLabel}>
            <ListItemText
              primary={`${ds.datasetLabel} (model: ${ds.modelID})`}
            />
            <IconButton onClick={() => handleDeleteDataset(ds.datasetLabel)}>
              <DeleteIcon />
            </IconButton>
          </ListItem>
        ))}
      </List>

      {/* Status and Progress */}
      <Typography>{statusMessage}</Typography>
      {isPredicting && (
        <Box my={2}>
          <LinearProgress variant="determinate" value={progress} />
          <Typography>{progress}%</Typography>
        </Box>
      )}

      {!window.crossOriginIsolated && (
        <Alert severity="warning" sx={{ mt: 2 }}>
          Unable to use multiple cpu cores - notify the site owner
        </Alert>
      )}

      {selectedFile ? (
        <PredictionsSankey datasetLabel={selectedFile.name} />
      ) : null}

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
