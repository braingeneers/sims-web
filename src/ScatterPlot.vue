<template>
  <div>
    <!-- Dataset visibility controls -->
    <div class="control-panel mb-3">
      <v-btn-toggle v-model="datasetVisibility" mandatory>
        <v-btn value="train" :disabled="!hasTrainData">Train</v-btn>
        <v-btn value="test" :disabled="!hasTestData">Test</v-btn>
        <v-btn value="both" :disabled="!hasTrainData || !hasTestData">Both</v-btn>
      </v-btn-toggle>

      <!-- Class visibility controls -->
      <div class="class-controls mt-2">
        <v-btn size="small" @click="showAllClasses" class="mr-2">Show All</v-btn>
        <v-btn size="small" @click="hideAllClasses">Hide All</v-btn>
      </div>
    </div>

    <!-- Chart container -->
    <div ref="chartContainer" style="width: 100%; height: 450px"></div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted, watch, shallowRef, nextTick, computed } from 'vue'
import * as echarts from 'echarts/core'
import { ScatterChart, ScatterSeriesOption } from 'echarts/charts'
import {
  TooltipComponent,
  TooltipComponentOption,
  GridComponent,
  GridComponentOption,
  LegendComponent,
  LegendComponentOption,
  DataZoomComponent,
  DataZoomComponentOption,
  ToolboxComponent,
  ToolboxComponentOption,
} from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'

// Combine option types
type ECOption = echarts.ComposeOption<
  | ScatterSeriesOption
  | TooltipComponentOption
  | GridComponentOption
  | LegendComponentOption // Add this
  | DataZoomComponentOption
  | ToolboxComponentOption
>

// Register necessary components
echarts.use([
  ScatterChart,
  TooltipComponent,
  GridComponent,
  LegendComponent, // Add this back
  DataZoomComponent,
  ToolboxComponent,
  CanvasRenderer,
])

// Props definition
interface Props {
  trainMappings: number[][]
  trainLabelPairs: number[][]
  testMappings: number[][]
  testLabelPairs: number[][]
  classNames: string[]
  themeName: 'light' | 'dark'
}

const props = withDefaults(defineProps<Props>(), {
  trainMappings: () => [],
  trainLabelPairs: () => [],
  testMappings: () => [],
  testLabelPairs: () => [],
})

// Expose methods to parent components
defineExpose({
  clearData,
  setData,
  addData,
})

// Chart refs and state
const chartContainer = ref<HTMLElement | null>(null)
const chartInstance = shallowRef<echarts.ECharts | null>(null)
const datasetVisibility = ref('both') // 'train', 'test', or 'both'
const hiddenClassIndices = ref<Set<number>>(new Set())

// Computed properties for data state
const hasTrainData = computed(
  () => props.trainMappings.length > 0 && props.trainLabelPairs.length > 0,
)
const hasTestData = computed(() => props.testMappings.length > 0 && props.testLabelPairs.length > 0)

// Add these after the other computed properties
const trainDataExtents = computed(() => {
  if (!trainData.value.length) return null

  return trainData.value.reduce(
    (acc, point) => ({
      xMin: Math.min(acc.xMin, point[0]),
      xMax: Math.max(acc.xMax, point[0]),
      yMin: Math.min(acc.yMin, point[1]),
      yMax: Math.max(acc.yMax, point[1]),
    }),
    {
      xMin: Infinity,
      xMax: -Infinity,
      yMin: Infinity,
      yMax: -Infinity,
    },
  )
})

// Internal data storage for incremental updates
const trainData = ref<number[][]>([])
const testData = ref<number[][]>([])

// Custom color palette
const customColors = [
  // first 12 colours generated with:
  // RColorBrewer::brewer.pal(n = 12, name = "Paired")
  '#A6CEE3',
  '#1F78B4',
  '#B2DF8A',
  '#33A02C',
  '#FB9A99',
  '#E31A1C',
  '#FDBF6F',
  '#FF7F00',
  '#CAB2D6',
  '#6A3D9A',
  '#FFFF99',
  '#B15928',
  // vivid interlude
  '#1ff8ff', // a bright blue
  // next 8 colours generated with:
  // RColorBrewer::brewer.pal(n = 8, "Dark2")
  '#1B9E77',
  '#D95F02',
  '#7570B3',
  '#E7298A',
  '#66A61E',
  '#E6AB02',
  '#A6761D',
  '#666666',
  // list below generated with iwanthue: all colours soft kmeans 20
  '#4b6a53',
  '#b249d5',
  '#7edc45',
  '#5c47b8',
  '#cfd251',
  '#ff69b4',
  '#69c86c',
  '#cd3e50',
  '#83d5af',
  '#da6130',
  '#5e79b2',
  '#c29545',
  '#532a5a',
  '#5f7b35',
  '#c497cf',
  '#773a27',
  '#7cb9cb',
  '#594e50',
  '#d3c4a8',
  '#c17e7f',
]

// Data preparation functions
function prepareData(mappings: number[][], labelPairs: number[][]) {
  if (!mappings || !labelPairs || mappings.length !== labelPairs.length) {
    console.warn('ScatterPlot: Mismatched or missing mappings/labelPairs.')
    return []
  }

  return mappings
    .map((coord, index) => {
      const classIndex = labelPairs[index]?.[0]
      if (typeof classIndex !== 'number') return null

      return [
        coord[0], // x
        coord[1], // y
        classIndex, // class index for coloring
      ]
    })
    .filter((item) => item !== null) as number[][]
}

// Initialize internal data state from props
function initializeDataFromProps() {
  if (hasTrainData.value) {
    trainData.value = prepareData(props.trainMappings, props.trainLabelPairs)
  }

  if (hasTestData.value) {
    testData.value = prepareData(props.testMappings, props.testLabelPairs)
  }
}

// Chart initialization
function initChart() {
  if (chartContainer.value) {
    chartInstance.value?.dispose()
    console.log(`ScatterPlot: Initializing chart with theme: ${props.themeName}`)

    chartInstance.value = echarts.init(chartContainer.value, props.themeName)
    chartInstance.value.resize()

    if (resizeObserver && chartContainer.value) {
      resizeObserver.observe(chartContainer.value)
    }
  } else {
    console.error('ScatterPlot: Chart container not found during init.')
  }
}

// Update chart with current data and visibility settings
function updateChart() {
  if (!chartInstance.value) {
    if (chartContainer.value) {
      initChart()
    } else {
      return
    }
  }

  // Prepare series based on visibility setting
  const series: any[] = []

  // Add training data series if visible
  if (
    (datasetVisibility.value === 'train' || datasetVisibility.value === 'both') &&
    trainData.value.length > 0
  ) {
    // Group training data by class
    const dataByClass = trainData.value.reduce(
      (acc, point) => {
        const classIndex = point[2] as number
        if (!acc[classIndex]) acc[classIndex] = []
        acc[classIndex].push([point[0], point[1]])
        return acc
      },
      {} as Record<number, number[][]>,
    )

    // Create a series for each class
    Object.entries(dataByClass).forEach(([classIndex, points]) => {
      const seriesName = props.classNames[parseInt(classIndex)] // Use class name instead of index
      series.push({
        name: seriesName, // Use actual class name as series name
        type: 'scatter',
        symbolSize: 4,
        data: points.map((point) => [...point, parseInt(classIndex)]), // Keep class index for coloring
        emphasis: {
          focus: 'series',
          scale: 1.5,
        },
        itemStyle: {
          color: customColors[parseInt(classIndex) % customColors.length],
          opacity: datasetVisibility.value === 'both' ? 0.15 : 0.8,
        },
        z: 1,
      })
    })
  }

  // Add test data series if visible
  if (
    (datasetVisibility.value === 'test' || datasetVisibility.value === 'both') &&
    testData.value.length > 0
  ) {
    // Group test data by class
    const dataByClass = testData.value.reduce(
      (acc, point) => {
        const classIndex = point[2] as number
        if (!acc[classIndex]) acc[classIndex] = []
        acc[classIndex].push([point[0], point[1]])
        return acc
      },
      {} as Record<number, number[][]>,
    )

    // Create a series for each class - using same naming as training data
    Object.entries(dataByClass).forEach(([classIndex, points]) => {
      const seriesName = props.classNames[parseInt(classIndex)] // Use class name instead of index
      series.push({
        name: seriesName, // Use actual class name as series name
        type: 'scatter',
        symbolSize: 5,
        data: points.map((point) => [...point, parseInt(classIndex)]), // Keep class index for coloring
        emphasis: {
          focus: 'series',
          scale: 1.5,
        },
        itemStyle: {
          color: customColors[parseInt(classIndex) % customColors.length],
          opacity: 0.8,
        },
        z: 2,
      })
    })
  }
  // Chart options
  const option: ECOption = {
    color: customColors,
    tooltip: {
      trigger: 'item',
      formatter: (params: any) => {
        if (!params.value || params.value.length < 3) return ''

        const classIndex = params.value[2]
        const className = props.classNames[classIndex] ?? 'Unknown'
        const coords = params.value
        const dataset = params.seriesName

        return `${dataset}<br/>Class: ${className}<br/>Coords: (${coords[0].toFixed(2)}, ${coords[1].toFixed(2)})`
      },
    },
    grid: {
      left: '3%',
      right: '20%', // Increased to make room for legend
      bottom: '3%',
      top: '3%',
      containLabel: false,
    },
    legend: {
      type: 'scroll',
      orient: 'vertical',
      right: 10,
      top: 'middle',
      itemWidth: 15,
      itemHeight: 10,
      textStyle: {
        fontSize: 11,
        overflow: 'truncate',
        width: 100,
      },
      selected: Object.fromEntries(
        props.classNames.map((name, index) => [name, !hiddenClassIndices.value.has(index)]),
      ),
      data: props.classNames,
    },
    xAxis: {
      type: 'value',
      show: false,
      scale: true,
      // Set fixed min/max from training data
      min: trainDataExtents.value?.xMin,
      max: trainDataExtents.value?.xMax,
    },
    yAxis: {
      type: 'value',
      show: false,
      scale: true,
      // Set fixed min/max from training data
      min: trainDataExtents.value?.yMin,
      max: trainDataExtents.value?.yMax,
    },
    dataZoom: [
      {
        type: 'inside',
        xAxisIndex: 0,
        filterMode: 'empty',
        // Disable zooming if we don't have training data extents
        disabled: !trainDataExtents.value,
      },
      {
        type: 'inside',
        yAxisIndex: 0,
        filterMode: 'empty',
        disabled: !trainDataExtents.value,
      },
    ],
    series: series.map((s) => ({
      ...s,
      itemStyle: {
        ...s.itemStyle,
        color: (params: any) => {
          const classIndex = params.value[2]
          return customColors[classIndex % customColors.length]
        },
      },
    })),
  }

  // Apply options to chart
  chartInstance.value?.setOption(option, true)
}

// Class visibility methods
function showAllClasses() {
  hiddenClassIndices.value.clear()
  updateChart()
}

function hideAllClasses() {
  // Create a set with all class indices
  hiddenClassIndices.value = new Set(Array.from({ length: props.classNames.length }, (_, i) => i))
  updateChart()
}

function toggleClass(index: number) {
  if (hiddenClassIndices.value.has(index)) {
    hiddenClassIndices.value.delete(index)
  } else {
    hiddenClassIndices.value.add(index)
  }
  updateChart()
}

// Add a legend select event handler
function handleLegendSelect(event: any) {
  const classIndex = props.classNames.indexOf(event.name)
  if (classIndex !== -1) {
    toggleClass(classIndex)
  }
}

// Exposed API methods for parent components
function clearData(type: 'train' | 'test' | 'all' = 'all') {
  if (type === 'train' || type === 'all') {
    trainData.value = []
  }
  if (type === 'test' || type === 'all') {
    testData.value = []
  }
  updateChart()
}

function setData(type: 'train' | 'test', mappings: number[][], labelPairs: number[][]) {
  const preparedData = prepareData(mappings, labelPairs)
  if (type === 'train') {
    trainData.value = preparedData
  } else {
    testData.value = preparedData
  }
  updateChart()
}

function addData(type: 'train' | 'test', mappings: number[][], labelPairs: number[][]) {
  const preparedData = prepareData(mappings, labelPairs)
  if (type === 'train') {
    trainData.value = [...trainData.value, ...preparedData]
  } else {
    testData.value = [...testData.value, ...preparedData]
  }
  updateChart()
}

// ResizeObserver setup
let resizeObserver: ResizeObserver | null = null

// Lifecycle hooks
onMounted(() => {
  nextTick(() => {
    initializeDataFromProps()

    if (chartContainer.value) {
      console.log('ScatterPlot: Component mounted, initializing chart.')
      initChart()
      updateChart()

      resizeObserver = new ResizeObserver(() => {
        chartInstance.value?.resize()
      })
      resizeObserver.observe(chartContainer.value)

      chartInstance.value?.on('legendselectchanged', handleLegendSelect)
    }
  })
})

onUnmounted(() => {
  console.log('ScatterPlot: Component unmounted, disposing chart.')
  resizeObserver?.disconnect()
  chartInstance.value?.off('legendselectchanged', handleLegendSelect)
  chartInstance.value?.dispose()
  chartInstance.value = null
})

// Watch for prop changes
watch(
  () => [props.trainMappings, props.trainLabelPairs, props.testMappings, props.testLabelPairs],
  () => {
    console.log('ScatterPlot: Data props changed, updating internal data.')
    initializeDataFromProps()
    nextTick(updateChart)
  },
  { deep: true },
)

watch(
  () => props.classNames,
  () => {
    console.log('ScatterPlot: Class names changed, updating chart.')
    nextTick(updateChart)
  },
)

// Watch for theme changes
watch(
  () => props.themeName,
  (newTheme, oldTheme) => {
    if (newTheme !== oldTheme && chartContainer.value) {
      console.log(`ScatterPlot: Theme changed to ${newTheme}. Re-initializing chart.`)
      nextTick(() => {
        initChart()
        updateChart()
      })
    }
  },
)

// Watch for visibility changes
watch(
  () => datasetVisibility.value,
  () => {
    console.log(`ScatterPlot: Dataset visibility changed to ${datasetVisibility.value}`)
    updateChart()
  },
)
</script>

<style scoped>
.control-panel {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

/* Make sure the chart container fills available space */
:deep(.echarts) {
  width: 100% !important;
  height: 100% !important;
}
</style>
