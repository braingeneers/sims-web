<template>
  <v-app>
    <v-app-bar>
      <v-toolbar-title>Cell Dimension</v-toolbar-title>
      <v-spacer></v-spacer>
      <v-btn @click="openFileDialog = true">
        <v-icon>mdi-folder-open</v-icon>
      </v-btn>
      <v-btn @click="toggleTheme">
        <v-icon>mdi-theme-light-dark</v-icon>
      </v-btn>
    </v-app-bar>
    <v-main>
      <div>Cell Stray</div>
    </v-main>
    <v-dialog v-model="openFileDialog" width="500">
      <v-card>
        <v-card-title>Open File</v-card-title>
        <v-card-text>
          <v-text-field
            v-model="selectedFile"
            label="Selected File"
            readonly
            @click="fileInput?.click()"
          ></v-text-field>
          <input
            ref="fileInput"
            type="file"
            accept=".h5ad"
            style="display: none;"
            @change="handleFileSelect"
          />
          <v-select
            v-model="selectedModel"
            :items="modelItems"
            label="Select Model"
            item-title="title"
            item-value="id"
            :loading="loadingModels"
            :disabled="loadingModels"
          ></v-select>
        </v-card-text>
        <v-card-actions>
          <v-progress-linear
            v-model="predictionProgress"
            height="25"
            color="primary"
          >
            <template v-slot:default>
              <strong>{{ Math.ceil(predictionProgress) }}%</strong>
            </template>
          </v-progress-linear>
          <v-spacer></v-spacer>
          <v-btn @click="openFileDialog = false">Close</v-btn>
          <v-btn
            color="primary"
            @click="handlePredict"
            :disabled="!selectedModel || !selectedFile"
          >
            Predict
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
  </v-app>
</template>

<script setup lang="ts">
import { useTheme } from 'vuetify'
import { ref, onMounted, onUnmounted } from 'vue'

interface Model {
  id: string
  title: string
  description?: string
  version?: string
  metadata?: Record<string, unknown>
}

const theme = useTheme()
const openFileDialog = ref(false)
const loadingModels = ref(true)
const selectedModel = ref<string>('')
const models = ref<Record<string, Model>>({})
const modelItems = ref<Model[]>([])
const selectedFile = ref('sample.h5ad')
const fileInput = ref<HTMLInputElement | null>(null)
const predictionProgress = ref(0)
const worker = ref<Worker | null>(null)

async function loadModels() {
  try {
    loadingModels.value = true
    worker.value = new Worker(new URL('./worker.js', import.meta.url))
    
    worker.value.onmessage = (event) => {
      if (event.data.type === 'predictionProgress') {
        predictionProgress.value = (event.data.countFinished / event.data.totalToProcess) * 100
      }
    }
    const modelsResponse = await fetch('/models/models.txt')
    const modelsText = await modelsResponse.text()
    const modelIds = modelsText.trim().split('\n')

    const modelPromises = modelIds.map(async (id) => {
      const response = await fetch(`/models/${id}.json`)
      const modelData = await response.json()
      return { id, ...modelData }
    })

    const modelList = await Promise.all(modelPromises)
    models.value = Object.fromEntries(modelList.map(model => [model.id, model]))
    modelItems.value = modelList
  } catch (error) {
    console.error('Error loading models:', error)
  } finally {
    loadingModels.value = false
  }
}

function toggleTheme() {
  theme.global.name.value = theme.global.current.value.dark ? 'light' : 'dark'
}

function handleFileSelect(event: Event) {
  const target = event.target as HTMLInputElement
  const file = target.files?.[0]
  if (file) {
    selectedFile.value = file.name
  }
}

function handlePredict() {
  if (!selectedModel.value || !selectedFile.value) return
  
  worker.value?.postMessage({
    type: 'startPrediction',
    modelID: selectedModel.value,
    modelURL: '/models',
    h5File: selectedFile.value,
    cellRangePercent: 100
  })
}

onMounted(() => {
  loadModels()
})

onUnmounted(() => {
  worker.value?.terminate()
})
</script>

<style>
@import '@mdi/font/css/materialdesignicons.min.css';

:root {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
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
