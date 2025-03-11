<template>
  <v-app>
    <v-app-bar>
      <v-toolbar-title>Cell Space</v-toolbar-title>
      <v-spacer></v-spacer>

      <!-- File Selector -->
      <v-select
        v-model="selectedFile"
        :items="fileOptions"
        label="H5AD File"
        variant="underlined"
        class="mx-2"
        style="max-width: 200px"
      ></v-select>

      <!-- First Worker Selector -->
      <v-select
        v-model="selectedFileWorker"
        :items="fileWorkerOptions"
        label="File Worker"
        variant="underlined"
        class="mx-2"
        style="max-width: 200px"
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
              indeterminate
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

const theme = useTheme()
const isProcessing = ref(false)
const currentStatus = ref('')

// Available files and workers
const fileOptions = ref(['sample.h5ad', 'pbmc.h5ad', 'brain.h5ad'])
const fileWorkerOptions = ref(['H5AD File Loader'])
const analysisWorkerOptions = ref(['Average Calculator', 'Min/Max Calculator'])

// Selected options
const selectedFile = ref('sample.h5ad')
const selectedFileWorker = ref('H5AD File Loader')
const selectedAnalysisWorker = ref('Average Calculator')

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

function initializeWorkers() {
  // Create file worker
  fileWorker.value = new Worker(new URL('./workers/fileWorker.js', import.meta.url))
  fileWorker.value.onmessage = handleFileWorkerMessage

  // Create processing worker
  processingWorker.value = new Worker(new URL('./workers/processingWorker.js', import.meta.url))
  processingWorker.value.onmessage = handleProcessingWorkerMessage
}

function runPipeline() {
  // Reset previous results
  fileStats.value = null
  analysisResults.value = []
  isProcessing.value = true
  currentStatus.value = 'Starting pipeline...'

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
  fileWorker.value?.postMessage({
    type: 'processFile',
    file: selectedFile.value,
    batchSize: batchSize.value,
  })
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
  } else if (type === 'error') {
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

onMounted(() => {
  initializeWorkers()
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
