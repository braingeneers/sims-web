<template>
  <v-app>
    <v-app-bar>
      <v-toolbar-title>Cell Space</v-toolbar-title>
      <v-spacer></v-spacer>

      <!-- File Selector -->
      <v-file-input
        v-model="selectedFile"
        accept=".h5ad"
        label="Select H5AD File"
        variant="underlined"
        class="mx-2"
        style="max-width: 250px"
        hide-details
        truncate-length="15"
        @update:model-value="handleFileSelected"
      ></v-file-input>

      <!-- First Worker Selector -->
      <v-select
        v-model="selectedFileWorker"
        :items="fileWorkerOptions"
        label="File Worker"
        variant="underlined"
        class="mx-2"
        style="max-width: 200px"
        item-title="title"
        item-value="value"
      ></v-select>

      <!-- Second Worker Selector -->
      <v-select
        v-model="selectedAnalysisWorker"
        :items="analysisWorkerOptions"
        label="Analysis Worker"
        variant="underlined"
        class="mx-2"
        style="max-width: 200px"
      ></v-select>

      <!-- Run Button -->
      <v-btn color="primary" class="mx-2" @click="runPipeline" :loading="isProcessing"> Run </v-btn>

      <v-btn @click="toggleTheme">
        <v-icon>mdi-theme-light-dark</v-icon>
      </v-btn>
    </v-app-bar>

    <v-main>
      <v-container fluid>
        <!-- Status Display -->
        <v-card class="mb-4">
          <v-card-title>Pipeline Status</v-card-title>
          <v-card-text>
            <v-progress-linear
              v-if="isProcessing"
              :model-value="predictionProgress"
              color="primary"
              height="10"
            ></v-progress-linear>
            <p v-if="currentStatus">{{ currentStatus }}</p>
          </v-card-text>
        </v-card>

        <!-- File Information Display -->
        <v-card v-if="fileStats" class="mb-4">
          <v-card-title>File Information</v-card-title>
          <v-card-text>
            <p><strong>Cells:</strong> {{ fileStats.numCells }}</p>
            <p><strong>Genes:</strong> {{ fileStats.numGenes }}</p>
          </v-card-text>
        </v-card>

        <!-- Analysis Results Display -->
        <v-card v-if="analysisResults.length > 0" class="mb-4">
          <v-card-title>Analysis Results</v-card-title>
          <v-card-text>
            <v-list>
              <v-list-item v-for="(result, index) in analysisResults" :key="index">
                <v-list-item-title>{{ result.type }} Result</v-list-item-title>
                <v-list-item-subtitle>{{ result.summary }}</v-list-item-subtitle>
              </v-list-item>
            </v-list>
          </v-card-text>
        </v-card>
      </v-container>
    </v-main>
  </v-app>
</template>

<script setup lang="ts">
import { useTheme } from 'vuetify'
import { ref, onMounted, onUnmounted } from 'vue'

import { openDB, IDBPDatabase } from 'idb'
import SIMSWorker from './workers/sims-worker?worker'

const theme = useTheme()
const isProcessing = ref(false)
const currentStatus = ref('')
const predictionProgress = ref(0)
const processingTime = ref(0)
const processingStartTime = ref(0)

// Model metadata type
interface ModelMetadata {
  title: string
  [key: string]: unknown
}

// Available files and workers
const fileWorkerOptions = ref<{ title: string; value: string }[]>([])
const analysisWorkerOptions = ref(['Average Calculator', 'Min/Max Calculator'])

// Selected options
const selectedFile = ref<File | null>(null)
const selectedFileWorker = ref('')
const selectedAnalysisWorker = ref('Average Calculator')

// Model metadata
const modelMetadata = ref<Record<string, ModelMetadata>>({})

// Fetch models and their metadata
async function fetchModels() {
  try {
    const modelsResponse = await fetch('/models/models.txt')
    const modelsText = await modelsResponse.text()
    const modelIds = modelsText.trim().split('\n')

    // Fetch metadata for each model
    const metadata: Record<string, ModelMetadata> = {}
    for (const modelId of modelIds) {
      const metadataResponse = await fetch(`/models/${modelId}.json`)
      metadata[modelId] = await metadataResponse.json()
    }

    modelMetadata.value = metadata
    fileWorkerOptions.value = modelIds.map((id) => ({
      title: metadata[id].title || id,
      value: id,
    }))

    // Set default selection if available
    if (fileWorkerOptions.value.length > 0) {
      selectedFileWorker.value = 'default'
    }
  } catch (error) {
    console.error('Error fetching models:', error)
    currentStatus.value = 'Error loading models'
  }
}

// Workers
const fileWorker = ref<Worker | null>(null)
const processingWorker = ref<Worker | null>(null)
const analysisWorker = ref<Worker | null>(null)

// Results
const fileStats = ref<{ numCells: number; numGenes: number } | null>(null)
const analysisResults = ref<
  Array<{
    type: string
    summary: string
    detailedResult?: Float32Array
    min?: Float32Array
    max?: Float32Array
  }>
>([])

// The batch size for processing
const batchSize = ref(100)

function toggleTheme() {
  theme.global.name.value = theme.global.current.value.dark ? 'light' : 'dark'
}

// IndexedDB
const DB_VERSION = 2

const resultsDB = ref<IDBPDatabase | null>(null)

async function loadDataset() {
  try {
    const db = await openDB('sims-web', DB_VERSION, {
      upgrade(db, oldVersion, newVersion, transaction) {
        // Case 1: No database - create it
        if (!db.objectStoreNames.contains('datasets')) {
          db.createObjectStore('datasets', {
            keyPath: 'datasetLabel',
            autoIncrement: true,
          })
        }

        // Case 2: Version 1 exists - clear it
        if (oldVersion === 1) {
          transaction.objectStore('datasets').clear()
          currentStatus.value = 'Database upgraded - previous results cleared'
        }
      },
    })

    // Case 3: Version 2 exists - load first dataset
    const keys = await db.getAllKeys('datasets')
    if (keys.length > 0) {
      resultsDB.value = await db.get('datasets', keys[0])
    }

    db.close()
  } catch (error) {
    console.error('Database error:', error)
    currentStatus.value = 'Error accessing database'
  }
}

function initializeWorkers() {
  // Create file worker
  fileWorker.value = new SIMSWorker()
  fileWorker.value.onmessage = handleFileWorkerMessage

  // Create processing worker
  processingWorker.value = new Worker(new URL('./workers/processingWorker.js', import.meta.url))
  processingWorker.value.onmessage = handleProcessingWorkerMessage
}

function runPipeline() {
  // Check if a file is selected
  if (!selectedFile.value) {
    currentStatus.value = 'Please select an H5AD file first'
    return
  }

  // Reset previous results
  fileStats.value = null
  analysisResults.value = []
  isProcessing.value = true
  currentStatus.value = 'Starting pipeline...'
  predictionProgress.value = 0
  processingStartTime.value = Date.now()

  // Initialize workers if needed
  if (!fileWorker.value || !processingWorker.value) {
    initializeWorkers()
  }

  // Create selected analysis worker based on selection
  if (analysisWorker.value) {
    analysisWorker.value.terminate()
  }

  if (selectedAnalysisWorker.value === 'Average Calculator') {
    analysisWorker.value = new Worker(new URL('./workers/averageWorker.js', import.meta.url))
    analysisWorker.value.onmessage = handleAnalysisWorkerMessage
  } else {
    analysisWorker.value = new Worker(new URL('./workers/minMaxWorker.js', import.meta.url))
    analysisWorker.value.onmessage = handleAnalysisWorkerMessage
  }

  // Start the pipeline by sending a message to the file worker
  const modelURL = `${window.location.protocol}//${window.location.host}/models`
  fileWorker.value?.postMessage({
    type: 'startPrediction',
    modelID: selectedFileWorker.value,
    modelURL: modelURL,
    h5File: selectedFile.value,
    cellRangePercent: 25,
  })
}

function handleFileSelected(file: File) {
  selectedFile.value = file
}

function handleFileWorkerMessage(event: MessageEvent) {
  const { type, message } = event.data

  if (type === 'status') {
    currentStatus.value = message
  } else if (type === 'fileStats') {
    fileStats.value = {
      numCells: event.data.numCells,
      numGenes: event.data.numGenes,
    }
    currentStatus.value = message
  } else if (type === 'batchData') {
    // Pass the batch data to the processing worker
    processingWorker.value?.postMessage({
      type: 'processBatch',
      batch: event.data.batch,
      batchSize: event.data.batchSize,
      numCells: event.data.numCells,
      numGenes: event.data.numGenes,
    })
    currentStatus.value = message
  } else if (type === 'predictionProgress') {
    // Update progress based on countFinished and totalToProcess
    const { countFinished, totalToProcess } = event.data
    predictionProgress.value = (countFinished / totalToProcess) * 100
    currentStatus.value = `Processing: ${countFinished} of ${totalToProcess} complete (${Math.round(predictionProgress.value)}%)`
  } else if (type === 'finishedPrediction') {
    // Calculate processing time
    processingTime.value = (Date.now() - processingStartTime.value) / 1000

    // Reset UI state
    isProcessing.value = false
    predictionProgress.value = 0
    currentStatus.value = ''

    // Add processing result to analysis results
    analysisResults.value.push({
      type: 'Prediction',
      summary: `Processed ${event.data.totalProcessed} items in ${processingTime.value.toFixed(2)} seconds`,
    })
  } else if (type === 'predictionError') {
    currentStatus.value = message
    isProcessing.value = false
  }
}

function handleProcessingWorkerMessage(event: MessageEvent) {
  const { type, message } = event.data

  if (type === 'status') {
    currentStatus.value = message
  } else if (type === 'batchProcessed') {
    // Pass the processed data to the selected analysis worker
    if (selectedAnalysisWorker.value === 'Average Calculator') {
      analysisWorker.value?.postMessage({
        type: 'computeAverage',
        batch: event.data.batch,
        batchSize: event.data.batchSize,
        numCells: event.data.numCells,
        numGenes: event.data.numGenes,
      })
    } else {
      analysisWorker.value?.postMessage({
        type: 'computeMinMax',
        batch: event.data.batch,
        batchSize: event.data.batchSize,
        numCells: event.data.numCells,
        numGenes: event.data.numGenes,
      })
    }
    currentStatus.value = message
  } else if (type === 'error') {
    currentStatus.value = message
    isProcessing.value = false
  }
}

function handleAnalysisWorkerMessage(event: MessageEvent) {
  const { type, message } = event.data

  if (type === 'status') {
    currentStatus.value = message
  } else if (type === 'analysisResult') {
    analysisResults.value.push(event.data.result)
    currentStatus.value = message
    isProcessing.value = false
  } else if (type === 'error') {
    currentStatus.value = message
    isProcessing.value = false
  }
}

// Fetch a sample file on application load
async function fetchSampleFile() {
  try {
    const sampleFileName = 'sample.h5ad'
    currentStatus.value = 'Loading sample file...'

    const response = await fetch(sampleFileName)
    const blob = await response.blob()
    const file = new File([blob], sampleFileName, { type: blob.type })

    selectedFile.value = file
    currentStatus.value = 'Sample file loaded'
    console.log('Sample File:', file)
  } catch (error) {
    console.error('Error loading sample file:', error)
    currentStatus.value = 'Error loading sample file'
  }
}

onMounted(() => {
  loadDataset()
  initializeWorkers()
  fetchModels()
  fetchSampleFile() // Load the sample file when the app mounts
})

onUnmounted(() => {
  // Clean up workers
  fileWorker.value?.terminate()
  processingWorker.value?.terminate()
  analysisWorker.value?.terminate()
})
</script>

<style>
@import '@mdi/font/css/materialdesignicons.min.css';

:root {
  font-family:
    -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans',
    'Helvetica Neue', sans-serif;
}

.v-application {
  font-family: inherit;
  line-height: 1.6;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

.v-toolbar-title {
  font-weight: 600;
}
</style>
