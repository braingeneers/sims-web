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
  LegendComponent,
  LegendComponentOption,
} from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'

// Combine option types
type ECOption = echarts.ComposeOption<
  | ScatterSeriesOption
  | TooltipComponentOption
  | GridComponentOption
  | VisualMapComponentOption
  | LegendComponentOption
>

// Register necessary components
echarts.use([
  ScatterChart,
  TooltipComponent,
  GridComponent,
  VisualMapComponent,
  LegendComponent,
  CanvasRenderer,
])

// Props definition
interface Props {
  mappings: number[][]
  labelPairs: number[][]
  classNames: string[]
  themeName: 'light' | 'dark' // +++ Add theme prop
}
const props = defineProps<Props>()

const chartContainer = ref<HTMLElement | null>(null)
const chartInstance = shallowRef<echarts.ECharts | null>(null)

const prepareChartData = () => {
  // ... (no changes needed here)
  if (!props.mappings || !props.labelPairs || props.mappings.length !== props.labelPairs.length) {
    console.warn('ScatterPlot: Mismatched or missing mappings/labelPairs.')
    return []
  }
  return props.mappings
    .map((coord, index) => {
      const groundTruthIndex = props.labelPairs[index]?.[0]
      if (typeof groundTruthIndex !== 'number') {
        console.warn(`ScatterPlot: Missing ground truth index at index ${index}`)
        return null
      }
      return [
        coord[0], // x
        coord[1], // y
        groundTruthIndex, // ground truth class index
      ]
    })
    .filter((item) => item !== null) as number[][]
}

// Function to initialize or re-initialize the chart
const initChart = () => {
  if (chartContainer.value) {
    // Dispose the old instance if it exists
    chartInstance.value?.dispose()
    console.log(`ScatterPlot: Initializing chart with theme: ${props.themeName}`)
    // Initialize with the theme name
    chartInstance.value = echarts.init(chartContainer.value, props.themeName) // <-- Use theme name here
    // Add resize listener for the new instance
    chartInstance.value.resize() // Initial resize call
    if (resizeObserver && chartContainer.value) {
      // Ensure observer is observing the correct container (should be the same)
      // resizeObserver.unobserve(chartContainer.value); // Might not be needed if container ref doesn't change
      resizeObserver.observe(chartContainer.value)
    }
  } else {
    console.error('ScatterPlot: Chart container not found during init.')
  }
}

const updateChart = () => {
  // Ensure instance exists (it should after initChart)
  if (!chartInstance.value || !props.classNames.length) {
    console.log('ScatterPlot: Chart instance or classNames not ready for update.')
    // Attempt to initialize if container exists but instance doesn't (e.g., initial load)
    if (chartContainer.value && !chartInstance.value) {
      console.log('ScatterPlot: Instance missing, attempting initialization.')
      initChart()
      // If initChart failed, chartInstance might still be null
      if (!chartInstance.value) return
    } else if (!chartInstance.value) {
      return // Exit if still no instance
    }
  }

  const chartData = prepareChartData()
  if (!chartData.length) {
    chartInstance.value?.clear()
    console.log('ScatterPlot: No data to display.')
    return
  }

  // --- Define your custom color palette ---
  // From https://github.com/david-barnett/microViz/blob/main/R/distinct_palette.R
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
    // "#FDFF00", // lemon (clashes with #FFFF99 on some screens)
    // "#00FF00", # lime (indistinguishable from bright blue on some screens)
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
    // with a couple of arbitrary tweaks by me
    '#4b6a53',
    '#b249d5',
    '#7edc45',
    '#5c47b8',
    '#cfd251',
    '#ff69b4', // hotpink
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

  // --- Chart Options ---
  // (No changes needed in the options object itself,
  // ECharts applies theme colors automatically)
  const option: ECOption = {
    color: customColors, // Use custom colors
    tooltip: {
      trigger: 'item',
      formatter: (params: any) => {
        if (!params.value || params.value.length < 3) return ''
        const classIndex = params.value[2]
        const className = props.classNames[classIndex] ?? 'Unknown'
        const coords = params.value
        return `Class: ${className} (Index: ${classIndex})<br/>Coords: (${coords[0].toFixed(2)}, ${coords[1].toFixed(2)})`
      },
    },
    grid: {
      left: '1%',
      right: '10%',
      bottom: '1%',
      top: '1%',
      containLabel: false,
    },
    xAxis: {
      show: false,
      type: 'value',
      splitLine: { show: false },
    },
    yAxis: {
      show: false,
      type: 'value',
      splitLine: { show: false },
    },
    visualMap: {
      type: 'piecewise',
      orient: 'vertical',
      right: 10,
      top: 'center',
      min: 0,
      max: props.classNames.length - 1,
      dimension: 2,
      pieces: props.classNames.map((name, index) => ({
        value: index,
        label: name.length > 25 ? name.slice(0, 22) + '...' : name,
        color: customColors[index % customColors.length], // Use custom color

        // Color is handled by the theme or default ECharts palette or colors if specified
      })),
      textStyle: {
        fontSize: 10,
        // Text color will be adapted by the theme
      },
      itemWidth: 15,
      itemHeight: 10,
      precision: 0,
      realtime: false,
      hoverLink: false,
    },
    series: [
      {
        name: 'Cell Types (Ground Truth)',
        type: 'scatter',
        symbolSize: 4,
        data: chartData,
        emphasis: {
          focus: 'series',
          scale: 1.5,
        },
        // large: true,
        // largeThreshold: 2000
      },
    ],
  }
  console.log('ScatterPlot: Setting options.')
  chartInstance.value.setOption(option, true)
}

// Resize observer
let resizeObserver: ResizeObserver | null = null

onMounted(() => {
  nextTick(() => {
    if (chartContainer.value) {
      console.log('ScatterPlot: Component mounted, initializing chart.')
      initChart() // Initialize with the current theme
      updateChart() // Apply options

      resizeObserver = new ResizeObserver(() => {
        // Use a debounce mechanism if resize events fire too rapidly
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

// Watch for prop changes to update the chart data
watch(
  () => [props.mappings, props.labelPairs, props.classNames],
  () => {
    console.log('ScatterPlot: Data props changed, updating chart.')
    nextTick(updateChart) // Update options/data
  },
  { deep: true },
)

// +++ Watch for theme changes +++
watch(
  () => props.themeName,
  (newTheme, oldTheme) => {
    if (newTheme !== oldTheme && chartContainer.value) {
      console.log(`ScatterPlot: Theme changed to ${newTheme}. Re-initializing chart.`)
      // Re-initialize with the new theme and re-apply options
      nextTick(() => {
        initChart()
        updateChart()
      })
    }
  },
)
</script>

<style scoped>
/* ... */
</style>
