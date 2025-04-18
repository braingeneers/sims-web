<template>
  <v-app>
    <v-app-bar dark fixed>
      <v-toolbar-title>Cell Space</v-toolbar-title>

      <!-- File Selector -->
      <v-file-input
        v-model="selectedFile"
        accept=".h5ad"
        variant="underlined"
        style="max-width: 25%"
        prepend-icon="mdi-dna"
        hint="You must select a .h5ad file"
        @update:model-value="handleFileSelected"
      >
        <template v-if="selectedFile" #selection>
          {{
            selectedFile.name.length > 30
              ? selectedFile.name.slice(0, 25) + '...'
              : selectedFile.name
          }}</template
        ></v-file-input
      >

      <!-- Labeling Background Dataset-->
      <v-select
        v-model="selectedPredictWorker"
        :items="predictWorkerOptions"
        variant="underlined"
        style="max-width: 25%"
        prepend-icon="mdi-label-multiple"
        item-title="title"
        item-value="value"
      ></v-select>

      <!-- Second Worker Selector -->
      <v-select
        v-model="selectedClusterWorker"
        :items="clusterWorkerOptions"
        prepend-icon="mdi-scatter-plot"
        variant="underlined"
        style="max-width: 25%"
      ></v-select>

      <!-- Run Button -->
      <v-app-bar-nav-icon
        data-cy="run-button"
        color="primary"
        @click="runPipeline"
        :loading="isProcessing"
        icon="mdi-play"
      >
      </v-app-bar-nav-icon>

      <!-- Stop Button -->
      <v-app-bar-nav-icon color="error" @click="handleStop" icon="mdi-stop"></v-app-bar-nav-icon>
      <v-app-bar-nav-icon @click="toggleTheme">
        <v-icon>mdi-theme-light-dark</v-icon>
      </v-app-bar-nav-icon>
      <template v-if="isProcessing" #extension>
        <!-- Status Display -->
        <v-card class="mb-" style="width: 100%" :flat="true">
          <v-card-text>
            <v-progress-linear
              v-if="isProcessing"
              :model-value="processingProgress"
              color="primary"
              height="4"
            ></v-progress-linear>
            <p v-if="currentStatus">{{ currentStatus }}</p>
          </v-card-text>
        </v-card>
      </template></v-app-bar
    >

    <v-main>
      <v-container fluid>
        <!-- Analysis Results Display -->
        <v-card v-if="analysisResults.length > 0" class="mb-4">
          <v-card-title>Analysis Progress</v-card-title>
          <v-card-text>
            <v-timeline v-if="analysisResults.length > 0" density="comfortable">
              <v-timeline-item
                v-for="(result, index) in analysisResults"
                :key="index"
                :dot-color="result.type === 'Prediction' ? 'primary' : 'success'"
                size="small"
              >
                <template v-slot:opposite>
                  <strong>{{ result.type }}</strong>
                </template>
                <div>{{ result.summary }}</div>
              </v-timeline-item>
            </v-timeline>
            <p v-else>No analysis results yet. Select a file and click the Run button to begin.</p>
          </v-card-text>
        </v-card>

        <!-- File Information Display -->
        <v-card v-if="resultsDB" class="mb-4">
          <v-card-title>{{ resultsDB.datasetLabel }}</v-card-title>
          <v-card-text data-cy="results">
            <strong>Cells:</strong> {{ resultsDB.cellNames.length }} <strong>Genes:</strong>
            {{ resultsDB.genes.length }}
          </v-card-text>
        </v-card>

        <!-- Predictions Table -->
        <v-card v-if="resultsDB" class="mb-4">
          <v-card-title>Predictions</v-card-title>
          <v-card-text>
            <predictions-table
              :cell-names="resultsDB.cellNames"
              :predictions="resultsDB.predictions"
              :probabilities="resultsDB.probabilities"
              :cell-type-names="resultsDB.cellTypeNames"
              :coordinates="resultsDB.coords"
            ></predictions-table>
          </v-card-text>
        </v-card>
      </v-container>
    </v-main>
  </v-app>
</template>

<script setup lang="ts">
import { useTheme } from 'vuetify'
import { ref, onMounted, onUnmounted } from 'vue'

import { openDB } from 'idb'
import SIMSWorker from './workers/sims-worker.ts?worker'
import UMAPWorker from './workers/umap-worker.ts?worker'

import PredictionsTable from './PredictionsTable.vue'

const theme = useTheme()
const isProcessing = ref(false)
const currentStatus = ref('')
const processingProgress = ref(0)
const processingTime = ref(0)
const processingStartTime = ref(0)

// Model metadata type
interface ModelMetadata {
  title: string
  submitter: string
  [key: string]: unknown
}

// TypeScript interface for dataset
interface Dataset {
  datasetLabel: string
  cellNames: string[]
  cellTypes: number[]
  cellTypeNames: string[]
  encodings: Float32Array[]
  predictions: number[][]
  probabilities: Float32Array[]
  coords: number[]
  genes: string[]
  overallTopGenes: number[]
  topGeneIndicesByClass: number[][]
}

// Available files and workers
const predictWorkerOptions = ref<{ title: string; value: string }[]>([])
const clusterWorkerOptions = ref(['UMAP', 'PUMAP', 'HDBSCAN'])

// Selected options
const selectedFile = ref<File | null>(null)
const selectedPredictWorker = ref('default')
const selectedClusterWorker = ref('UMAP')

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
    predictWorkerOptions.value = modelIds.map((id) => ({
      title: metadata[id].title || id,
      value: id,
    }))
  } catch (error) {
    console.error('Error fetching models:', error)
    currentStatus.value = 'Error loading models'
  }
}

// Workers
const predictWorker = ref<Worker | null>(null)
const clusterWorker = ref<Worker | null>(null)

// Results
const analysisResults = ref<
  Array<{
    type: string
    summary: string
    detailedResult?: Float32Array
    min?: Float32Array
    max?: Float32Array
  }>
>([])

function toggleTheme() {
  theme.global.name.value = theme.global.current.value.dark ? 'light' : 'dark'
}

// IndexedDB
const DB_VERSION = 2

const resultsDB = ref<Dataset | null>(null)

async function loadDataset() {
  try {
    const db = await openDB('sims-web', DB_VERSION, {
      upgrade(db, oldVersion, _newVersion, transaction) {
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

async function clearResults() {
  try {
    const db = await openDB('sims-web', DB_VERSION)
    await db.clear('datasets')
    db.close()
    // Clear UI dataset if present
    resultsDB.value = null
  } catch (error) {
    console.error('Error clearing database:', error)
    currentStatus.value = 'Error clearing previous results'
  }
}

function initializeWorkers() {
  // Create SIMS worker
  predictWorker.value = new SIMSWorker()
  predictWorker.value.onmessage = handlePredictWorkerMessage

  // Create UMAP worker
  clusterWorker.value = new UMAPWorker()
  clusterWorker.value.onmessage = handleClusterWorkerMessage
}

async function runPipeline() {
  // Check if a file is selected
  if (!selectedFile.value) {
    currentStatus.value = 'Please select an H5AD file first'
    return
  }

  // Clear previous results from database and UI
  await clearResults()
  analysisResults.value = []

  isProcessing.value = true
  currentStatus.value = 'Starting pipeline...'
  processingProgress.value = 0
  processingStartTime.value = Date.now()

  // Initialize workers if needed
  if (!predictWorker.value || !clusterWorker.value) {
    initializeWorkers()
  }

  // Start the pipeline by sending a message to the file worker
  const modelURL = `${window.location.protocol}//${window.location.host}/models`
  predictWorker.value?.postMessage({
    type: 'startPrediction',
    modelID: selectedPredictWorker.value,
    modelURL: modelURL,
    h5File: selectedFile.value,
    cellRangePercent: 25,
  })
}

function handleStop() {
  // Terminate workers
  predictWorker.value?.terminate()
  clusterWorker.value?.terminate()
  // Reinitialize workers
  initializeWorkers()
  isProcessing.value = false

  clearResults()
}

function handleFileSelected(files: File | File[]) {
  const file = Array.isArray(files) ? files[0] : files
  if (file) {
    selectedFile.value = file
  } else {
    selectedFile.value = null
  }
}

function handlePredictWorkerMessage(event: MessageEvent) {
  const { type, message } = event.data

  if (type === 'status') {
    currentStatus.value = message
  } else if (type === 'processingProgress') {
    // Update progress based on countFinished and totalToProcess
    const { countFinished, totalToProcess } = event.data
    processingProgress.value = (countFinished / totalToProcess) * 100
    currentStatus.value = `Processing: ${countFinished} of ${totalToProcess} complete (${Math.round(processingProgress.value)}%)`
  } else if (type === 'finishedPrediction') {
    // Add processing result to analysis results
    analysisResults.value.push({
      type: 'Prediction',
      summary: `Processed ${event.data.totalProcessed} items in ${((Date.now() - processingStartTime.value) / 1000).toFixed(2)} seconds`,
    })

    // Reset progress bar for UMAP
    processingProgress.value = 0
    processingStartTime.value = Date.now()

    // Start UMAP computation
    currentStatus.value = 'Starting UMAP computation...'
    clusterWorker.value?.postMessage({
      type: 'startUMAP',
      datasetLabel: event.data.datasetLabel,
    })
  } else if (type === 'predictionError') {
    currentStatus.value = message
    isProcessing.value = false
  }
}

function handleClusterWorkerMessage(event: MessageEvent) {
  const { type, message } = event.data

  if (type === 'status') {
    currentStatus.value = message
  } else if (type === 'processingProgress') {
    // Update progress based on countFinished and totalToProcess
    const { countFinished, totalToProcess } = event.data
    processingProgress.value = (countFinished / totalToProcess) * 100
    currentStatus.value = `UMAP: ${countFinished} of ${totalToProcess} complete (${Math.round(processingProgress.value)}%)`
  } else if (type === 'finishedUMAP') {
    // Calculate processing time
    processingTime.value = (Date.now() - processingStartTime.value) / 1000

    // Reset UI state
    isProcessing.value = false
    processingProgress.value = 0
    currentStatus.value = ''

    // Add processing result to analysis results
    analysisResults.value.push({
      type: 'UMAP',
      summary: `Computed UMAP coordinates in ${processingTime.value.toFixed(2)} seconds`,
    })
    loadDataset()
  } else if (type === 'umapError') {
    currentStatus.value = `UMAP Error: ${event.data.error}`
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
  predictWorker.value?.terminate()
  clusterWorker.value?.terminate()
})
</script>

<style></style>
