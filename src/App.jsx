import { useState, useEffect } from "react";
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
} from "@mui/material";

import { MuiFileInput } from "mui-file-input";

import NewspaperIcon from "@mui/icons-material/Newspaper";
import GitHubIcon from "@mui/icons-material/GitHub";

import { PredictionsTable } from "./PredictionsTable";
import { PredictionsPlot } from "./PredictionsPlot";
import { PredictionsSankey } from "./PredictionsSankey";

import { openDB } from "idb";

import SIMSWorker from "./worker?worker";

function App() {
  const [modelInfoList, setModelInfoList] = useState([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [selectedFile, setSelectedFile] = useState(null);
  const [cellRangePercent, setCellRangePercent] = useState(25);
  const [statusMessage, setStatusMessage] = useState("");
  const [progress, setProgress] = useState(0);
  const [isPredicting, setIsPredicting] = useState(false);
  const [workerInstance, setWorkerInstance] = useState(null);
  const [dataset, setDataset] = useState(null);

  const sitePath =
    window.location.origin +
    window.location.pathname.slice(
      0,
      window.location.pathname.lastIndexOf("/")
    );

  useEffect(() => {
    fetchModels();
    fetchSampleFile();
    loadDataset();
  }, []);

  const DB_VERSION = 2;

  async function loadDataset() {
    try {
      const db = await openDB("sims-web", DB_VERSION, {
        upgrade(db, oldVersion, newVersion, transaction) {
          // Case 1: No database - create it
          if (!db.objectStoreNames.contains("datasets")) {
            db.createObjectStore("datasets", {
              keyPath: "datasetLabel",
              autoIncrement: true,
            });
          }

          // Case 2: Version 1 exists - clear it
          if (oldVersion === 1) {
            transaction.objectStore("datasets").clear();
            setStatusMessage("Database upgraded - previous results cleared");
          }
        },
      });

      // Case 3: Version 2 exists - load first dataset
      const keys = await db.getAllKeys("datasets");
      if (keys.length > 0) {
        setDataset(await db.get("datasets", keys[0]));
      }

      db.close();
    } catch (error) {
      console.error("Database error:", error);
      setStatusMessage("Error accessing database");
    }
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
        case "finished":
          setStatusMessage(
            `Processed ${evt.data.totalProcessed} of ${
              evt.data.totalNumCells
            } cells in ${evt.data.elapsedTime?.toFixed(2)} minutes`
          );
          setIsPredicting(false);
          loadDataset();
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
    setDataset(null);

    // Delete all datasets
    const db = await openDB("sims-web");
    const tx = db.transaction("datasets", "readwrite");
    const store = tx.objectStore("datasets");
    store.clear();
    await tx.done;

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

      {dataset && (
        <Typography variant="h5" sx={{ mt: 4 }}>
          Results for {dataset.datasetLabel}
        </Typography>
      )}

      {/* Layout for sankey + top genes + scatter plot */}
      {dataset && (
        <Box display="flex" mt={4}>
          <Box width="40%" mr={4}>
            {/* <PredictionsSankey datasetLabel={dataset.datasetLabel} /> */}
          </Box>
          <Box width="20%" mr={4}>
            <Typography variant="h6">Top Genes</Typography>
            <List dense>
              {dataset.overallTopGenes.map((geneIndex) => (
                <ListItem key={geneIndex}>
                  <ListItemText primary={`${dataset.genes[geneIndex]}`} />
                </ListItem>
              ))}
            </List>
          </Box>
          <Box width="40%" mr={4}>
            <PredictionsPlot
              width={450}
              height={450}
              labels={dataset.cellTypes}
              coordinates={dataset.coords}
            />
          </Box>
        </Box>
      )}

      {/* Container for table or results */}
      {dataset && (
        <PredictionsTable
          cellNames={dataset.cellNames}
          predictions={dataset.predictions}
          probabilities={dataset.probabilities}
          cellTypeNames={dataset.cellTypeNames}
          coordinates={dataset.coords}
        />
      )}
    </Container>
  );
}

export default App;
