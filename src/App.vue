<template>
  <v-app>
    <!-- Navigation Drawer with hamburger in its header -->
    <v-navigation-drawer v-model="drawerOpen" :temporary="$vuetify.display.smAndDown" app>
      <v-list>
        <v-list-item>
          <div class="d-flex align-center w-100">
            <v-app-bar-nav-icon @click="toggleDrawer" class="ml-n2"></v-app-bar-nav-icon>
            <v-list-item-title class="text-h6 mr-auto">SIMS Web</v-list-item-title>
            <v-btn
              variant="text"
              icon
              size="small"
              class="pa-0"
              style="min-width: 28px"
              href="https://www.cell.com/cell-genomics/abstract/S2666-979X(24)00165-4"
              target="_blank"
              aria-label="Cell Genomics Paper"
            >
              <v-icon>mdi-newspaper</v-icon>
            </v-btn>
            <v-btn
              icon
              size="small"
              variant="text"
              class="pa-0"
              style="min-width: 28px"
              href="https://github.com/braingeneers/sims-web"
              target="_blank"
              aria-label="GitHub Repository"
            >
              <v-icon>mdi-github</v-icon>
            </v-btn>
          </div>
        </v-list-item>
      </v-list>

      <v-divider></v-divider>

      <v-list density="compact" nav>
        <!-- File Selector -->
        <v-list-item>
          <template #prepend>
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

        <!-- Background Dataset and Model Selector -->
        <v-list-item>
          <template #prepend>
            <v-icon>mdi-label-multiple</v-icon>
          </template>
          <v-select
            v-model="selectedDataset"
            :items="datasetOptions"
            variant="plain"
            density="compact"
            hide-details
            item-title="title"
            item-value="value"
            class="mt-n2"
          ></v-select>
        </v-list-item>

        <v-divider class="my-2"></v-divider>

        <!-- Run Button -->
        <v-list-item @click="runPipeline" :disabled="isProcessing" data-cy="run-button">
          <template #prepend>
            <v-icon color="primary">mdi-play</v-icon>
          </template>
          <v-list-item-title>Run</v-list-item-title>
        </v-list-item>

        <!-- Stop Button -->
        <v-list-item @click="handleStop" :disabled="!isProcessing">
          <template #prepend>
            <v-icon color="error">mdi-stop</v-icon>
          </template>
          <v-list-item-title>Stop</v-list-item-title>
        </v-list-item>

        <v-divider class="my-2"></v-divider>

        <!-- Theme Toggle -->
        <v-list-item @click="toggleTheme">
          <template #prepend>
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
          <v-card-title>Welcome to SIMS Web</v-card-title>
          <v-card-text>
            <p>Use the side menu to select a file and run an analysis.</p>
            <p class="text-caption">
              A sample file has been automatically loaded. Just click the Run button in the sidebar
              to process it, or select an h5ad file from your computer.
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

        <!-- WebGL Scatter Plot Card -->
        <v-card
          v-if="trainMappings && cellTypeClasses.length > 0"
          class="mb-4"
          data-cy="webgl-scatter-plot-card"
        >
          <v-card-title class="text-subtitle-1">Reference Model Visualization (WebGL)</v-card-title>
          <v-card-subtitle> 2D projection of reference model data (train) </v-card-subtitle>
          <v-card-text>
            <scatter-plot-web-g-l
              :train-mappings="trainMappings || []"
              :class-names="cellTypeClasses"
              :theme-name="theme.global.name.value === 'dark' ? 'dark' : 'light'"
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

import SIMSWorker from './worker.ts?worker'

import CellTypeChart from './CellTypeChart.vue'
import ScatterPlotWebGL from './ScatterPlotWebGL.vue'
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
const datasetOptions = ref<{ title: string; value: string }[]>([])

// Selected options
const selectedFile = ref<File | null>(null)
const selectedDataset = ref('allen-celltypes+human-cortex+various-cortical-areas')

// Model metadata
const modelMetadata = ref<Record<string, ModelMetadata>>({})

// Add this new ref to store cell type classes
const cellTypeClasses = ref<string[]>([])
const trainMappings = ref<Float32Array | null>(null)

// Add this new ref

// Fetch models and their metadata
async function fetchModels() {
  try {
    const modelsResponse = await fetch('models/models.txt')
    const modelsText = await modelsResponse.text()
    const modelIds = modelsText.trim().split('\n')

    // Fetch metadata for each model
    const metadata: Record<string, ModelMetadata> = {}
    for (const modelId of modelIds) {
      const metadataResponse = await fetch(`models/${modelId}.json`)
      metadata[modelId] = await metadataResponse.json()
    }

    modelMetadata.value = metadata
    datasetOptions.value = modelIds.map((id) => ({
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
  const classesResponse = await fetch(`models/${modelId}.classes`)
  if (!classesResponse.ok) {
    throw new Error(`Failed to fetch classes: ${classesResponse.statusText}`)
  }
  const classesText = await classesResponse.text()
  const classes = classesText.trim().split('\n')
  cellTypeClasses.value = classes

  // Create and assign mappings variable directly
  // const mappings = ref<Float32Array>(new Float32Array())

  try {
    // Use fetch with appropriate headers to ensure correct binary transfer
    const mappingsResponse = await fetch(`models/${modelId}-mappings.bin`, {
      headers: {
        Accept: 'application/octet-stream',
      },
    })

    if (mappingsResponse.ok) {
      // Get the complete ArrayBuffer from the response
      const mappingsBuffer = await mappingsResponse.arrayBuffer()

      // Check buffer size to verify we got complete data
      console.log(`Received mappings buffer of ${mappingsBuffer.byteLength} bytes`)

      // Assign directly to the reference
      trainMappings.value = new Float32Array(mappingsBuffer)

      console.log(`Loaded ${trainMappings.value.length / 3} points from mappings.bin`)
    }
  } catch (error) {
    console.warn(`Error fetching mappings: ${error}`)
  }

  return {
    classes, // Array of class names
    trainMappings, // Float32Array with [x, y, classIndex] values
  }
}

// Workers
const predictWorker = ref<Worker | null>(null)

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
    } else {
      resultsDB.value = null // Clear if no data found
    }

    db.close()
  } catch (error) {
    console.error('Database error:', error)
    currentStatus.value = 'Error accessing database'
    resultsDB.value = null // Clear on error
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
  if (!predictWorker.value) {
    initializeWorkers()
  }

  // Start the pipeline by sending a message to the file worker
  const modelURL = `${window.location.href}models`
  predictWorker.value?.postMessage({
    type: 'startPrediction',
    modelID: selectedDataset.value,
    modelURL: modelURL,
    h5File: selectedFile.value,
    cellRangePercent: 100,
  })
}

function handleStop() {
  // Terminate workers
  predictWorker.value?.terminate()
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
    const { topKIndices, coords } = event.data
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

    // Calculate processing time
    processingTime.value = (Date.now() - processingStartTime.value) / 1000

    // Reset UI state
    isProcessing.value = false
    processingProgress.value = 0
    currentStatus.value = ''

    // Add processing result to analysis results
    analysisResults.value.push({
      type: 'Overall',
      summary: `Processed ${event.data.totalProcessed} items in ${processingTime.value.toFixed(2)} seconds`,
    })

    // Auto-collapse drawer when finished/
    // drawerOpen.value = false

    // Load the dataset
    loadDataset()
  } else if (type === 'predictionError') {
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

// Add a watcher for the selected predict worker
watch(selectedDataset, (newModelId) => {
  if (newModelId) {
    fetchCellTypeClasses(newModelId)
  }
})

watch(
  () => trainMappings.value,
  (newMappings) => {
    if (newMappings && newMappings.length > 0) {
      // The mappings loaded from the file are already in Float32Array format
      trainMappings.value = newMappings
      console.log(`Model data loaded with ${newMappings.length / 3} points`)
    }
  },
  { immediate: true },
)

onMounted(() => {
  loadDataset()
  initializeWorkers()
  fetchModels()
  fetchSampleFile() // Load the sample file when the app mounts

  // Also fetch cell types for the default model
  if (selectedDataset.value) {
    fetchCellTypeClasses(selectedDataset.value)
  }
})

onUnmounted(() => {
  // Clean up workers
  predictWorker.value?.terminate()
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
