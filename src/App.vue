<template>
  <v-app>
    <!-- Navigation Drawer with hamburger in its header -->
    <v-navigation-drawer v-model="drawerOpen" :temporary="$vuetify.display.smAndDown" app>
      <v-list>
        <v-list-item>
          <template v-slot:prepend>
            <v-app-bar-nav-icon @click="toggleDrawer"></v-app-bar-nav-icon>
          </template>
          <v-list-item-title class="text-h6">Cell Space</v-list-item-title>
          <template v-slot:append>
            <v-avatar size="small" color="primary" class="ml-2">
              <v-img
                src="https://raw.githubusercontent.com/vuetifyjs/vuetify-loader/next/packages/vuetify-loader/src/logo.svg"
              ></v-img>
            </v-avatar>
          </template>
        </v-list-item>
      </v-list>

      <v-divider></v-divider>

      <v-list density="compact" nav>
        <!-- File Selector -->
        <v-list-item>
          <template v-slot:prepend>
            <v-icon>mdi-dna</v-icon>
          </template>
          <v-list-item-title>Select File</v-list-item-title>
          <v-list-item-subtitle v-if="selectedFile">
            {{
              selectedFile.name.length > 20
                ? selectedFile.name.slice(0, 15) + '...'
                : selectedFile.name
            }}
          </v-list-item-subtitle>
          <template v-slot:append>
            <v-file-input
              v-model="selectedFile"
              accept=".h5ad"
              hide-details
              density="compact"
              variant="plain"
              class="file-input-hidden"
              @update:model-value="handleFileSelected"
            ></v-file-input>
          </template>
        </v-list-item>

        <!-- Labeling Background Dataset -->
        <v-list-item>
          <template v-slot:prepend>
            <v-icon>mdi-label-multiple</v-icon>
          </template>
          <v-select
            v-model="selectedPredictWorker"
            :items="predictWorkerOptions"
            variant="plain"
            density="compact"
            hide-details
            item-title="title"
            item-value="value"
            class="mt-n2"
          ></v-select>
        </v-list-item>

        <!-- Second Worker Selector -->
        <v-list-item>
          <template v-slot:prepend>
            <v-icon>mdi-scatter-plot</v-icon>
          </template>
          <v-select
            v-model="selectedClusterWorker"
            :items="clusterWorkerOptions"
            variant="plain"
            density="compact"
            hide-details
            class="mt-n2"
          ></v-select>
        </v-list-item>

        <v-divider class="my-2"></v-divider>

        <!-- Run Button -->
        <v-list-item @click="runPipeline" :disabled="isProcessing" data-cy="run-button">
          <template v-slot:prepend>
            <v-icon color="primary">mdi-play</v-icon>
          </template>
          <v-list-item-title>Run</v-list-item-title>
        </v-list-item>

        <!-- Stop Button -->
        <v-list-item @click="handleStop" :disabled="!isProcessing">
          <template v-slot:prepend>
            <v-icon color="error">mdi-stop</v-icon>
          </template>
          <v-list-item-title>Stop</v-list-item-title>
        </v-list-item>

        <v-divider class="my-2"></v-divider>

        <!-- Theme Toggle -->
        <v-list-item @click="toggleTheme">
          <template v-slot:prepend>
            <v-icon>mdi-theme-light-dark</v-icon>
          </template>
          <v-list-item-title>Toggle Theme</v-list-item-title>
        </v-list-item>
      </v-list>

      <!-- Processing Progress moved to the bottom -->
      <template v-slot:append>
        <div v-if="isProcessing" class="pa-2">
          <v-progress-linear
            :model-value="processingProgress"
            color="primary"
            height="4"
          ></v-progress-linear>
          <div class="text-caption mt-1">{{ currentStatus }}</div>
        </div>

        <!-- Analysis Progress Timeline in the drawer -->
        <div v-if="analysisResults.length > 0" class="pa-2 mt-auto">
          <v-divider class="mb-2"></v-divider>
          <div class="text-subtitle-2 mb-1">Analysis Progress</div>
          <v-timeline density="compact" class="analysis-timeline">
            <v-timeline-item
              v-for="(result, index) in analysisResults"
              :key="index"
              :dot-color="result.type === 'Prediction' ? 'primary' : 'success'"
              size="x-small"
              density="compact"
            >
              <div class="text-caption">
                <strong>{{ result.type }}:</strong> {{ result.summary }}
              </div>
            </v-timeline-item>
          </v-timeline>
        </div>
      </template>
    </v-navigation-drawer>

    <!-- Updated floating button to toggle drawer when it's closed -->
    <v-btn
      v-if="!drawerOpen"
      icon="mdi-menu"
      size="large"
      color="primary"
      style="position: fixed; top: 12px; left: 16px; z-index: 100;"
      @click="toggleDrawer"
      class="floating-menu-btn"
    ></v-btn>

    <v-main>
      <v-container fluid>
        <!-- Start with a welcome message when no analysis has been run -->
        <v-card v-if="!analysisResults.length && !resultsDB" class="mb-4">
          <v-card-title>Welcome to Cell Space</v-card-title>
          <v-card-text>
            <p>Use the side menu to select a file and run an analysis.</p>
            <p class="text-caption">
              A sample file has been automatically loaded. Click the Run button in the sidebar to
              process it.
            </p>
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

// Drawer state
const drawerOpen = ref(true)

// Replace drawerMini with toggleDrawer function
function toggleDrawer() {
  drawerOpen.value = !drawerOpen.value
}

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
    
    // Auto-collapse drawer when finished
    drawerOpen.value = false
    
    // Load the dataset
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

<style>
.file-input-hidden :deep(.v-field__input) {
  padding-top: 0;
}

.file-input-hidden :deep(.v-field__append-inner) {
  padding-top: 0;
}

/* Styling for the timeline in the drawer */
.analysis-timeline {
  max-height: 200px;
  overflow-y: auto;
}

.analysis-timeline :deep(.v-timeline-item__body) {
  padding: 4px 0;
}

/* Better positioning for the floating menu button */
.floating-menu-btn {
  box-shadow: 0 3px 5px -1px rgba(0,0,0,.2), 0 6px 10px 0 rgba(0,0,0,.14), 0 1px 18px 0 rgba(0,0,0,.12);
  margin: 0;
}

/* Ensure proper transitions */
.v-navigation-drawer {
  transition: transform 0.3s ease-in-out, width 0.3s ease-in-out;
}
</style>
