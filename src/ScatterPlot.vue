<template>
  <div ref="chartContainer" style="width: 100%; height: 450px"></div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted, watch, shallowRef, nextTick } from 'vue'
import * as echarts from 'echarts/core'
import { ScatterChart, ScatterSeriesOption } from 'echarts/charts'
import {
  TooltipComponent,
  TooltipComponentOption,
  GridComponent,
  GridComponentOption,
  VisualMapComponent,
  VisualMapComponentOption,
  LegendComponent, // Import LegendComponent
  LegendComponentOption,
} from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'

// Combine option types
type ECOption = echarts.ComposeOption<
  | ScatterSeriesOption
  | TooltipComponentOption
  | GridComponentOption
  | VisualMapComponentOption
  | LegendComponentOption // Add Legend option type
>

// Register necessary components
echarts.use([
  ScatterChart,
  TooltipComponent,
  GridComponent,
  VisualMapComponent,
  LegendComponent, // Register LegendComponent
  CanvasRenderer,
])

// Props definition
interface Props {
  mappings: number[][] // Array of [x, y]
  labelPairs: number[][] // Array of [groundTruth, prediction]
  classNames: string[] // Array of class names
}
const props = defineProps<Props>()

const chartContainer = ref<HTMLElement | null>(null)
const chartInstance = shallowRef<echarts.ECharts | null>(null)

const prepareChartData = () => {
  if (!props.mappings || !props.labelPairs || props.mappings.length !== props.labelPairs.length) {
    console.warn('ScatterPlot: Mismatched or missing mappings/labelPairs.')
    return []
  }
  // Combine mappings and the ground truth label index
  // Format: [x, y, groundTruthClassIndex]
  return props.mappings
    .map((coord, index) => {
      // Ensure labelPairs[index] and labelPairs[index][0] exist
      const groundTruthIndex = props.labelPairs[index]?.[0]
      if (typeof groundTruthIndex !== 'number') {
        console.warn(`ScatterPlot: Missing ground truth index at index ${index}`)
        // Handle missing data, e.g., assign a specific index or filter out
        return null // Or assign a default like [coord[0], coord[1], -1]
      }
      return [
        coord[0], // x
        coord[1], // y
        groundTruthIndex, // ground truth class index
      ]
    })
    .filter((item) => item !== null) as number[][] // Filter out null entries if any
}

const updateChart = () => {
  if (!chartContainer.value || !props.classNames.length) {
    console.log('ScatterPlot: Chart container or classNames not ready.')
    return
  }

  const chartData = prepareChartData()
  if (!chartData.length) {
    // Clear chart if no data
    chartInstance.value?.clear()
    console.log('ScatterPlot: No data to display.')
    return
  }

  if (!chartInstance.value) {
    chartInstance.value = echarts.init(chartContainer.value)
    console.log('ScatterPlot: ECharts instance initialized.')
  }

  const option: ECOption = {
    tooltip: {
      trigger: 'item',
      formatter: (params: any) => {
        // params.value should be [x, y, classIndex]
        if (!params.value || params.value.length < 3) return ''
        const classIndex = params.value[2]
        const className = props.classNames[classIndex] ?? 'Unknown'
        const coords = params.value
        return `Class: ${className} (Index: ${classIndex})<br/>Coords: (${coords[0].toFixed(2)}, ${coords[1].toFixed(2)})`
      },
    },
    grid: {
      left: '3%',
      right: '10%', // Increased space for visualMap
      bottom: '3%',
      containLabel: true,
    },
    xAxis: {
      type: 'value',
      splitLine: { show: false },
      axisLabel: { show: false }, // Hide axis labels for cleaner look
      axisTick: { show: false }, // Hide axis ticks
    },
    yAxis: {
      type: 'value',
      splitLine: { show: false },
      axisLabel: { show: false }, // Hide axis labels
      axisTick: { show: false }, // Hide axis ticks
    },
    visualMap: {
      type: 'piecewise',
      orient: 'vertical',
      right: 10,
      top: 'center',
      min: 0,
      max: props.classNames.length - 1,
      dimension: 2, // Map color based on the 3rd value in data (class index)
      pieces: props.classNames.map((name, index) => ({
        value: index,
        label: name.length > 25 ? name.slice(0, 22) + '...' : name, // Shorten long labels
        // ECharts assigns colors automatically
      })),
      textStyle: {
        fontSize: 10,
      },
      itemWidth: 15,
      itemHeight: 10,
      precision: 0, // Ensure integer matching for pieces
      realtime: false, // Update view only when interaction ends
      hoverLink: false, // Disable hover linkage for performance with many pieces
    },
    series: [
      {
        name: 'Cell Types (Ground Truth)',
        type: 'scatter',
        symbolSize: 4, // Smaller points for potentially large datasets
        data: chartData,
        emphasis: {
          focus: 'series',
          scale: 1.5, // Slightly enlarge points on hover
        },
        large: true, // Enable large-scale optimization
        largeThreshold: 2000, // Threshold to enable optimization
      },
    ],
  }
  console.log('ScatterPlot: Setting options.')
  chartInstance.value.setOption(option, true) // `true` to not merge options
}

// Resize observer
let resizeObserver: ResizeObserver | null = null
onMounted(() => {
  // Ensure the DOM element is ready before initializing
  nextTick(() => {
    if (chartContainer.value) {
      console.log('ScatterPlot: Component mounted, initializing chart.')
      updateChart() // Initial render

      resizeObserver = new ResizeObserver(() => {
        chartInstance.value?.resize()
      })
      resizeObserver.observe(chartContainer.value)
    } else {
      console.error('ScatterPlot: Chart container not found on mount.')
    }
  })
})

onUnmounted(() => {
  console.log('ScatterPlot: Component unmounted, disposing chart.')
  resizeObserver?.disconnect()
  chartInstance.value?.dispose()
  chartInstance.value = null
})

// Watch for prop changes to update the chart
watch(
  () => [props.mappings, props.labelPairs, props.classNames],
  () => {
    console.log('ScatterPlot: Props changed, updating chart.')
    // Use nextTick to ensure DOM updates (if any) are flushed before chart update
    nextTick(updateChart)
  },
  { deep: true }, // Use deep watch cautiously, might be performance intensive if props are large/complex
)
</script>

<style scoped>
/* Add any specific styles if needed */
div {
  border: 1px solid rgba(128, 128, 128, 0.2); /* Optional: subtle border */
  border-radius: 4px;
}
</style>
