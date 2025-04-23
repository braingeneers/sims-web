<!-- src/CellTypeChart.vue -->
<template>
  <v-card class="mb-4">
    <v-card-title class="d-flex align-center">
      <div>{{ chartTitle }}</div>
      <v-spacer></v-spacer>
      <v-chip v-if="totalItems > 0" class="ml-2" size="small">{{ totalItems }} Types</v-chip>
    </v-card-title>
    <v-card-text>
      <v-chart
        v-if="!isLoading && chartOptions"
        class="chart"
        :option="chartOptions"
        :theme="vuetifyTheme.global.current.value.dark ? 'dark' : 'light'"
        autoresize
      />
      <div v-else-if="isLoading" class="text-center pa-4">
        <v-progress-circular indeterminate color="primary"></v-progress-circular>
        <p class="mt-2 text-caption">Loading Chart Data...</p>
      </div>
      <div v-else class="text-center pa-4 text-caption">
        No cell type data available. Run an analysis.
      </div>
    </v-card-text>
  </v-card>
</template>

<script setup lang="ts">
import { ref, computed, watch, PropType } from 'vue'
import { useTheme } from 'vuetify'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { BarChart, LineChart } from 'echarts/charts' // LineChart for potential future use
import {
  TitleComponent,
  TooltipComponent,
  GridComponent,
  LegendComponent,
  DataZoomComponent,
  ToolboxComponent,
} from 'echarts/components'
import VChart from 'vue-echarts'
import type { EChartsOption } from 'echarts'

// --- ECharts Registration ---
use([
  CanvasRenderer,
  BarChart,
  LineChart,
  TitleComponent,
  TooltipComponent,
  GridComponent,
  LegendComponent,
  DataZoomComponent,
  ToolboxComponent,
])

// --- Props ---
interface FinalResults {
  cellTypeNames: string[]
  predictions: number[][] // Assuming predictions[i][0] is the top prediction index for cell i
  // Add other relevant fields from Dataset if needed
}

const props = defineProps({
  isProcessing: {
    type: Boolean,
    required: true,
  },
  dynamicCounts: {
    type: Object as PropType<Record<string, number>>, // Index (string) -> count
    required: true,
  },
  finalResults: {
    type: Object as PropType<FinalResults | null>,
    default: null,
  },
  // Pass the names corresponding to indices used in dynamicCounts and finalResults
  modelCellTypeNames: {
    type: Array as PropType<string[]>,
    required: true,
  },
})

// --- State ---
const vuetifyTheme = useTheme()
const chartOptions = ref<EChartsOption | null>(null)
const isLoading = ref(false) // To show loading state when calculating final counts

// --- Computed Properties ---
const chartTitle = computed(() =>
  props.isProcessing ? 'Live Cell Type Counts' : 'Final Cell Type Counts',
)

const totalItems = computed(() => chartData.value.categories.length)

const chartData = computed(() => {
  const categories: string[] = []
  const values: number[] = []
  let dataAvailable = false

  if (props.isProcessing) {
    // --- Dynamic Data (During Processing) ---
    const dynamicData: { name: string; value: number }[] = []
    if (props.dynamicCounts && props.modelCellTypeNames) {
      for (const indexStr in props.dynamicCounts) {
        const index = parseInt(indexStr, 10)
        const count = props.dynamicCounts[indexStr]
        // Ensure index is valid for the modelCellTypeNames array
        if (index >= 0 && index < props.modelCellTypeNames.length) {
          const name = props.modelCellTypeNames[index] || `Unknown Index ${index}`
          dynamicData.push({ name, value: count })
        } else {
          console.warn(`Index ${index} out of bounds for modelCellTypeNames`)
          dynamicData.push({ name: `Unknown Index ${index}`, value: count })
        }
      }
      // Sort by count descending
      dynamicData.sort((a, b) => b.value - a.value)
      categories.push(...dynamicData.map((item) => item.name))
      values.push(...dynamicData.map((item) => item.value))
      dataAvailable = dynamicData.length > 0
    }
  } else if (props.finalResults) {
    // --- Final Data (After Processing) ---
    const finalCounts: Record<string, number> = {}
    if (
      props.finalResults.predictions &&
      props.finalResults.cellTypeNames &&
      props.finalResults.cellTypeNames.length > 0 // Ensure names are loaded
    ) {
      props.finalResults.predictions.forEach((predictionArray) => {
        if (predictionArray && predictionArray.length > 0) {
          const topPredictionIndex = predictionArray[0]
          // Ensure index is valid for the final cellTypeNames array
          if (
            props.finalResults &&
            topPredictionIndex >= 0 &&
            topPredictionIndex < props.finalResults.cellTypeNames.length
          ) {
            const typeName = props.finalResults.cellTypeNames[topPredictionIndex]
            if (typeName) {
              finalCounts[typeName] = (finalCounts[typeName] || 0) + 1
            }
          } else {
            console.warn(
              `Prediction index ${topPredictionIndex} out of bounds for finalResults.cellTypeNames`,
            )
            const unknownKey = `Unknown Index ${topPredictionIndex}`
            finalCounts[unknownKey] = (finalCounts[unknownKey] || 0) + 1
          }
        }
      })

      // Convert to arrays for ECharts, sort by count descending
      const sortedEntries = Object.entries(finalCounts).sort(
        ([, countA], [, countB]) => countB - countA,
      )
      categories.push(...sortedEntries.map(([name]) => name))
      values.push(...sortedEntries.map(([, count]) => count))
      dataAvailable = sortedEntries.length > 0
    }
  }

  return { categories, values, dataAvailable }
})

// --- Watchers ---
watch(
  chartData,
  (newData) => {
    isLoading.value = false // Stop loading once data is processed
    if (newData.dataAvailable) {
      chartOptions.value = {
        tooltip: {
          trigger: 'axis',
          axisPointer: {
            type: 'shadow',
          },
        },
        grid: {
          left: '3%',
          right: '4%',
          bottom: '3%',
          containLabel: true,
        },
        xAxis: [
          {
            type: 'category',
            data: newData.categories,
            axisTick: {
              alignWithLabel: true,
            },
            axisLabel: {
              interval: 0, // Show all labels
              rotate: 45, // Rotate labels if they overlap
              fontSize: 10,
            },
          },
        ],
        yAxis: [
          {
            type: 'value',
            name: 'Count',
            nameLocation: 'middle',
            nameGap: 35, // Adjust gap as needed
          },
        ],
        dataZoom: [
          // Enable zooming and scrolling for many categories
          {
            type: 'inside',
            start: 0,
            end: newData.categories.length > 20 ? 50 : 100, // Show first 50% if many items
          },
          {
            start: 0,
            end: newData.categories.length > 20 ? 50 : 100,
            handleSize: '80%',
            height: 20, // Adjust height of the slider
            bottom: 20, // Position slider at the bottom
          },
        ],
        series: [
          {
            name: 'Cell Count',
            type: 'bar',
            barWidth: '60%',
            data: newData.values,
            itemStyle: {
              // Optional: Use Vuetify primary color
              // color: vuetifyTheme.global.current.value.colors.primary
            },
          },
        ],
        toolbox: {
          // Add toolbox for saving image etc.
          feature: {
            saveAsImage: { title: 'Save Image' },
          },
          right: 20,
          top: 0,
        },
      }
    } else {
      chartOptions.value = null // Clear chart if no data
    }
  },
  { immediate: true, deep: true }, // Deep watch dynamicCounts
)

// Show loading indicator when switching to final results if calculation takes time
watch(
  () => props.isProcessing,
  (processing) => {
    if (!processing && props.finalResults) {
      isLoading.value = true
      // Calculation happens in computed 'chartData', watcher above will set isLoading to false
    } else if (processing) {
      isLoading.value = false // No loading needed for dynamic updates
    }
  },
)

// Watch for theme changes
watch(
  () => vuetifyTheme.global.name.value,
  () => {
    // The theme prop on v-chart handles this automatically
    // If more complex theme logic is needed, update chartOptions here
  },
)
</script>

<style scoped>
.chart {
  height: 400px; /* Adjust height as needed */
  width: 100%;
}
.v-card-text {
  /* Ensure card text can contain the chart properly */
  overflow: hidden;
}
</style>
