<template>
  <v-app>
    <!-- Navigation Drawer with hamburger in its header -->
    <v-navigation-drawer v-model="drawerOpen" :temporary="$vuetify.display.smAndDown" app>
      <v-list>
        <v-list-item>
          <template v-slot:prepend>
            <v-app-bar-nav-icon @click="toggleDrawer" class="ml-n2"></v-app-bar-nav-icon>
          </template>
          <v-list-item-title class="text-h6">Cell Space</v-list-item-title>
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

      <!-- Bottom of left side bar -->
      <template v-slot:append>
        <!-- Processing Progress -->
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
      style="position: fixed; top: 12px; left: 2px; z-index: 100"
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

        <!-- Model Ground Truth Scatter Plot -->
        <!-- Show this plot if model data is loaded, regardless of processing or resultsDB -->
        <v-card
          v-if="modelMappings && modelLabelPairs && cellTypeClasses.length > 0"
          class="mb-4"
          data-cy="model-scatter-plot-card"
        >
          <v-card-title class="text-subtitle-1">Model Ground Truth Visualization</v-card-title>
          <v-card-subtitle>Reference distribution for {{ selectedPredictWorker }}</v-card-subtitle>
          <v-card-text>
            <scatter-plot
              :mappings="modelMappings"
              :label-pairs="modelLabelPairs"
              :class-names="cellTypeClasses"
              :theme-name="theme.global.name.value"
            />
          </v-card-text>
        </v-card>

        <!-- Show chart if processing OR if final results exist -->
        <cell-type-chart
          v-if="isProcessing || resultsDB"
          :is-processing="isProcessing"
          :dynamic-counts="labelCounts"
          :final-results="resultsDB"
          :model-cell-type-names="cellTypeClasses"
        />
        <!-- Optionally show model classes list if NOT processing and NO results yet -->
        <v-card v-else-if="cellTypeClasses.length > 0" class="mb-4">
          <v-card-title class="d-flex align-center">
            <div>Available Model Classes</div>
            <v-spacer></v-spacer>
            <v-chip class="ml-2" size="small">{{ cellTypeClasses.length }}</v-chip>
          </v-card-title>
          <v-card-text>
            <v-sheet class="cell-type-classes" elevation="0">
              <v-chip-group>
                <v-chip
                  v-for="(className, index) in cellTypeClasses"
                  :key="index"
                  :color="index % 2 === 0 ? 'primary' : 'secondary'"
                  variant="outlined"
                  size="small"
                  class="ma-1"
                >
                  {{ className }}
                </v-chip>
              </v-chip-group>
            </v-sheet>
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
import { ref, onMounted, onUnmounted, watch } from 'vue'

import { openDB } from 'idb'

import SIMSWorker from './workers/sims-worker.ts?worker'
import UMAPWorker from './workers/umap-worker.ts?worker'

import CellTypeChart from './CellTypeChart.vue'
import ScatterPlot from './ScatterPlot.vue'
import PredictionsTable from './PredictionsTable.vue'

// Drawer state
const drawerOpen = ref(true)
function toggleDrawer() {
  drawerOpen.value = !drawerOpen.value
}

const theme = useTheme()
const isProcessing = ref(false)
const currentStatus = ref('')
const processingProgress = ref(0)
const processingTime = ref(0)
const processingStartTime = ref(0)

const labelCounts = ref<Record<string, number>>({})

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
// const selectedPredictWorker = ref('default')
const selectedPredictWorker = ref('allen-celltypes+human-cortex+various-cortical-areas')
const selectedClusterWorker = ref('UMAP')

// Model metadata
const modelMetadata = ref<Record<string, ModelMetadata>>({})

// Add this new ref to store cell type classes
const cellTypeClasses = ref<string[]>([])
const modelMappings = ref<number[][] | null>(null)
const modelLabelPairs = ref<number[][] | null>(null)

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

// Fetch the model's classes, mappings, and label pairs
async function fetchCellTypeClasses(modelId: string) {
  // Fetch class names
  const classesResponse = await fetch(`/models/${modelId}.classes`)
  if (!classesResponse.ok) {
    throw new Error(`Failed to fetch classes: ${classesResponse.statusText}`)
  }
  const classesText = await classesResponse.text()
  const classes = classesText.trim().split('\n')
  cellTypeClasses.value = classes

  // Fetch 2D mappings (coordinates for visualization)
  let mappings = null
  let labelPairs = null

  try {
    // Use fetch with appropriate headers to ensure correct binary transfer
    const mappingsResponse = await fetch(`/models/${modelId}-mappings.bin`, {
      headers: {
        Accept: 'application/octet-stream',
      },
    })

    if (mappingsResponse.ok) {
      // Get the complete ArrayBuffer from the response
      const mappingsBuffer = await mappingsResponse.arrayBuffer()

      // Check buffer size to verify we got complete data
      console.log(`Received mappings buffer of ${mappingsBuffer.byteLength} bytes`)

      // Create a properly sized Float32Array view of the buffer
      // Each pair is 2 float32 values (8 bytes total per pair)
      const expectedPairs = Math.floor(mappingsBuffer.byteLength / 8)
      console.log(`Expected number of coordinate pairs: ${expectedPairs}`)

      const mappingsData = new Float32Array(mappingsBuffer)

      // Verify the array length
      console.log(`Float32Array length: ${mappingsData.length}, expected: ${expectedPairs * 2}`)

      // Convert flat array to array of [x, y] pairs
      mappings = []
      for (let i = 0; i < mappingsData.length; i += 2) {
        if (i + 1 < mappingsData.length) {
          // Ensure we don't go out of bounds
          mappings.push([mappingsData[i], mappingsData[i + 1]])
        }
      }

      modelMappings.value = mappings

      console.log(`Loaded ${mappings.length} 2D coordinates from mappings.bin`)
    }
  } catch (error) {
    console.warn(`Error fetching mappings: ${error.message}`)
  }

  // Fetch label pairs (ground truth and prediction)
  try {
    const labelsResponse = await fetch(`/models/${modelId}-labels.bin`, {
      headers: {
        Accept: 'application/octet-stream',
      },
    })

    if (labelsResponse.ok) {
      const labelsBuffer = await labelsResponse.arrayBuffer()

      // Check buffer size
      console.log(`Received labels buffer of ${labelsBuffer.byteLength} bytes`)

      // Each pair is 2 int32 values (8 bytes total per pair)
      const expectedPairs = Math.floor(labelsBuffer.byteLength / 8)
      console.log(`Expected number of label pairs: ${expectedPairs}`)

      // Create a view of the buffer as Int32Array
      const labelsData = new Int32Array(labelsBuffer)

      // Verify the array length
      console.log(`Int32Array length: ${labelsData.length}, expected: ${expectedPairs * 2}`)

      // Convert flat array to array of [groundTruth, prediction] pairs
      labelPairs = []
      for (let i = 0; i < labelsData.length; i += 2) {
        if (i + 1 < labelsData.length) {
          // Ensure we don't go out of bounds
          labelPairs.push([labelsData[i], labelsData[i + 1]])
        }
      }

      modelLabelPairs.value = labelPairs

      console.log(`Loaded ${labelPairs.length} label pairs from labels.bin`)
    }
  } catch (error) {
    console.warn(`Error fetching label pairs: ${error.message}`)
  }

  return {
    classes, // Array of class names
    mappings, // Array of [x, y] coordinates for each cell
    labelPairs, // Array of [groundTruth, prediction] class indices for each cell
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

  labelCounts.value = {}

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
    cellRangePercent: 100,
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
  } else if (type === 'predictionOutput') {
    const { topKIndices } = event.data
    // More idiomatic TypeScript approach
    topKIndices.cpuData.forEach((index: number) => {
      const key: string = index.toString()
      labelCounts.value[key] = (labelCounts.value[key] ?? 0) + 1
    })
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

    // Auto-collapse drawer when finished/
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

// Add a watcher for the selected predict worker
watch(selectedPredictWorker, (newModelId) => {
  if (newModelId) {
    fetchCellTypeClasses(newModelId)
  }
})

onMounted(() => {
  loadDataset()
  initializeWorkers()
  fetchModels()
  fetchSampleFile() // Load the sample file when the app mounts

  // Also fetch cell types for the default model
  if (selectedPredictWorker.value) {
    fetchCellTypeClasses(selectedPredictWorker.value)
  }
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
</style>
