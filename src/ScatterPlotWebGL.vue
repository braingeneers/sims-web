<template>
  <div class="chart-container">
    <div class="controls">
      <button :class="{ active: showBoth }" @click="setVisibility('both')">Both</button>
      <button :class="{ active: showTrainOnly }" @click="setVisibility('train')">Training</button>
      <button :class="{ active: showTestOnly }" @click="setVisibility('test')">Test</button>
    </div>
    <div ref="chartContainer" style="width: 100%; height: 600px"></div>
  </div>
</template>

<script lang="ts">
import { defineComponent, onMounted, onUnmounted, ref, watch, PropType } from 'vue'
import * as echarts from 'echarts/core'
import { ScatterGLChart } from 'echarts-gl/charts'

import { GridComponent, VisualMapComponent, LegendComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'

// Register necessary components
echarts.use([ScatterGLChart, GridComponent, VisualMapComponent, LegendComponent, CanvasRenderer])

export default defineComponent({
  name: 'ScatterPlotWebGL',
  props: {
    // Expecting a Float32Array with [x,y,label index] format
    trainMappings: {
      type: Float32Array,
      required: true,
    },
    // Expecting a Float32Array with [x,y,label index] format
    testMappings: {
      type: Float32Array,
      required: false,
      default: null,
    },
    classNames: {
      type: Array,
      required: true,
      default: () => [],
    },
    themeName: {
      type: String,
      default: 'light',
    },
  },
  setup(props) {
    const chartContainer = ref<HTMLElement | null>(null)
    let chart: echarts.ECharts | null = null
    const showBoth = ref(true)
    const showTrainOnly = ref(false)
    const showTestOnly = ref(false)

    const pieces = props.classNames.map((name, index) => {
      const hue = (index * 137.5) % 360 // Consistent color generation
      return {
        value: index, // The class index
        label: name as string, // The class name
        color: `hsl(${hue}, 70%, 50%)`,
      }
    })

    // ECharts configuration
    const option = {
      xAxis: {
        show: false,
      },
      yAxis: {
        show: false,
      },
      visualMap: {
        type: 'piecewise',
        dimension: 2,
        pieces: pieces,
        outOfRange: {
          symbolSize: 0,
        },
        show: true,
        orient: 'vertical',
        left: 10,  // Change from right to left
        top: 70,   // Position below the buttons
        itemWidth: 15,
        itemHeight: 15,
        textStyle: {
          fontSize: 12,
        },
      },
      series: [
        {
          name: 'Reference',
          type: 'scatterGL', // WebGL renderer
          data: props.trainMappings, // Use props.trainMappings instead of data
          dimensions: ['x', 'y', 'class'],
          symbolSize: 1,
          itemStyle: {
            opacity: 0.1,
          },
        },
        {
          name: 'Predictions',
          type: 'scatterGL', // WebGL renderer
          data: props.testMappings,
          dimensions: ['x', 'y', 'class'],
          symbolSize: 5,
        },
      ],
    }

    const updateSeriesVisibility = () => {
      if (!chart) return

      const updatedOption = {
        series: [
          {
            name: 'Reference',
            data: showBoth.value || showTrainOnly.value ? props.trainMappings : [],
            symbolSize: 1,
            itemStyle: {
              opacity: 0.07,
            },
          },
          {
            name: 'Predictions',
            data: showBoth.value || showTestOnly.value ? props.testMappings : [],
            symbolSize: 5,
          },
        ],
      }
      chart.setOption(updatedOption, { notMerge: false })
    }

    const setVisibility = (mode: 'both' | 'train' | 'test') => {
      showBoth.value = mode === 'both'
      showTrainOnly.value = mode === 'train'
      showTestOnly.value = mode === 'test'
      updateSeriesVisibility()
    }

    // Function to initialize or reinitialize the chart
    const initChart = () => {
      if (!chartContainer.value) return

      // Dispose of previous chart instance if it exists
      if (chart) {
        chart.dispose()
      }

      // Create new chart with current theme
      chart = echarts.init(chartContainer.value, props.themeName)
      chart.setOption(option)
    }

    onMounted(() => {
      initChart()
    })

    watch(
      () => [props.trainMappings, props.testMappings, props.classNames],
      updateSeriesVisibility,
      { deep: true }, // Use deep watch if props themselves might mutate, though Float32Array replacement is fine
    )

    // Watch for theme changes and re-initialize chart when it changes
    watch(
      () => props.themeName,
      () => {
        initChart()
      },
    )

    onUnmounted(() => {
      if (chart) {
        chart.dispose()
        chart = null
      }
    })

    return {
      chartContainer,
      showBoth,
      showTrainOnly,
      showTestOnly,
      setVisibility,
    }
  },
})
</script>

<style scoped>
.chart-container {
  position: relative;
  width: 100%;
  height: 100%;
}

.controls {
  position: absolute;
  top: 10px;
  left: 10px; /* Change from right to left */
  z-index: 1;
  display: flex;
  gap: 0.5rem;
}

button {
  padding: 0.5rem 1rem;
  border: 1px solid #ccc;
  border-radius: 4px;
  background: white;
  cursor: pointer;
  font-size: 12px;
}

button.active {
  background: #4caf50;
  color: white;
  border-color: #45a049;
}

button:hover {
  background: #f0f0f0;
}

button.active:hover {
  background: #45a049;
}
</style>
