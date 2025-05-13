<template>
  <div class="chart-container">
    <div class="controls-header">
      <div class="legend-controls">
        <button @click="toggleAllClasses(true)">Show All</button>
        <button @click="toggleAllClasses(false)">Hide All</button>
      </div>
      <div
        class="visibility-controls"
        v-if="trainMappings && trainMappings.length > 0 && testMappings && testMappings.length > 0"
      >
        <button :class="{ active: showBoth }" @click="setVisibility('both')">Both</button>
        <button :class="{ active: showTrainOnly }" @click="setVisibility('train')">Training</button>
        <button :class="{ active: showTestOnly }" @click="setVisibility('test')">Test</button>
      </div>
    </div>

    <div ref="chartContainer" style="width: 100%; height: 600px"></div>
  </div>
</template>

<script lang="ts">
import { defineComponent, onMounted, onUnmounted, ref, watch, computed, PropType } from 'vue'
import * as echarts from 'echarts'
// import * as echarts from 'echarts/core'
import { ScatterGLChart } from 'echarts-gl/charts'

import { GridComponent, VisualMapComponent, LegendComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'

// Register necessary components
echarts.use([ScatterGLChart, GridComponent, VisualMapComponent, LegendComponent, CanvasRenderer])

// Define a Vue component
export default defineComponent({
  name: 'ScatterPlotWebGL',
  props: {
    // Expecting a Float32Array with [x,y,label index] format
    trainMappings: {
      type: Object as PropType<Float32Array>,
      required: true,
    },
    // Expecting a Float32Array with [x,y,label index] format
    testMappings: {
      type: Object as PropType<Float32Array>,
      required: false,
    },
    classNames: {
      type: Array,
      required: true,
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
    // const hiddenClasses = ref<Set<number>>(new Set())

    const pieces = computed(() => {
      return props.classNames.map((name, index) => {
        const hue = (index * 137.5) % 360 // Consistent color generation
        return {
          value: index, // The class index
          label: name as string, // The class name
          color: `hsl(${hue}, 70%, 50%)`,
        }
      })
    })

    // Calculate bounds from training only so extent doesn't shift as test mappings are added
    const calculateBounds = (data: Float32Array) => {
      let minX = Infinity
      let maxX = -Infinity
      let minY = Infinity
      let maxY = -Infinity

      for (let i = 0; i < data.length; i += 3) {
        const x = data[i]
        const y = data[i + 1]

        minX = Math.min(minX, x)
        maxX = Math.max(maxX, x)
        minY = Math.min(minY, y)
        maxY = Math.max(maxY, y)
      }

      // Add 5% padding to the bounds
      const xPadding = (maxX - minX) * 0.05
      const yPadding = (maxY - minY) * 0.05

      return {
        xMin: minX - xPadding,
        xMax: maxX + xPadding,
        yMin: minY - yPadding,
        yMax: maxY + yPadding,
      }
    }

    // Function to initialize or reinitialize the chart
    const getChartOption = () => {
      const currentBounds = calculateBounds(props.trainMappings)
      return {
        xAxis: {
          show: false,
          min: currentBounds.xMin,
          max: currentBounds.xMax,
        },
        yAxis: {
          show: false,
          min: currentBounds.yMin,
          max: currentBounds.yMax,
        },
        visualMap: {
          type: 'piecewise',
          dimension: 2, // Corresponds to the 3rd item in data array [x, y, classIndex]
          pieces: pieces.value,
          outOfRange: {
            symbolSize: 0, // Points not in any piece (or for deselected pieces) will be hidden
          },
          show: true,
          orient: 'vertical',
          left: 10,
          top: 10,
          itemWidth: 15,
          itemHeight: 15,
          textStyle: {
            fontSize: 12,
          },
          // `selected` will be controlled by updateChart
        },
        series: [
          {
            name: 'Reference',
            type: 'scatterGL',
            data: [], // Initial data set by updateChart
            dimensions: ['x', 'y', 'class'],
            symbolSize: 1,
            itemStyle: {
              opacity: 0.1,
            },
          },
          {
            name: 'Predictions',
            type: 'scatterGL',
            data: [], // Initial data set by updateChart
            dimensions: ['x', 'y', 'class'],
            symbolSize: 5,
          },
        ],
      }
    }

    const initChart = () => {
      if (!chartContainer.value) return
      if (chart) {
        chart.dispose()
      }
      chart = echarts.init(chartContainer.value, props.themeName)
      chart.setOption(getChartOption())

      updateChart() // Load initial data
    }

    const updateChart = () => {
      if (!chart) return

      // Add type assertion for the getOption result
      const options = chart.getOption() as echarts.EChartsOption
      const visualMap = Array.isArray(options.visualMap) ? options.visualMap[0] : options.visualMap

      // Type assertion for the visualMap to specify it's a PiecewiseVisualMapOption
      const currentSelection =
        (visualMap as echarts.PiecewiseVisualMapComponentOption)?.selected || {}

      let trainDataToShow: Float32Array | number[][] = new Float32Array()
      if ((showBoth.value || showTrainOnly.value) && props.trainMappings) {
        trainDataToShow = props.trainMappings // Pass full data; visualMap will filter
      }

      let testDataToShow: Float32Array | number[][] = new Float32Array()
      if ((showBoth.value || showTestOnly.value) && props.testMappings) {
        testDataToShow = props.testMappings // Pass full data; visualMap will filter
      }

      chart.setOption({
        visualMap: {
          selected: currentSelection,
        },
        series: [
          { name: 'Reference', data: trainDataToShow },
          { name: 'Predictions', data: testDataToShow },
        ],
      })
    }

    const setVisibility = (mode: 'both' | 'train' | 'test') => {
      showBoth.value = mode === 'both'
      showTrainOnly.value = mode === 'train'
      showTestOnly.value = mode === 'test'
      updateChart()
    }

    const toggleAllClasses = (show: boolean) => {
      if (!chart) return

      const options = chart.getOption() as echarts.EChartsOption
      const visualMap = (
        Array.isArray(options.visualMap) ? options.visualMap[0] : options.visualMap
      ) as echarts.PiecewiseVisualMapComponentOption
      const currentSelection = visualMap?.selected || {}

      for (let i = 0; i < visualMap.pieces!.length; i++) {
        currentSelection[i] = show
      }

      chart.setOption({
        visualMap: {
          selected: currentSelection,
        },
      })
    }

    onMounted(() => {
      // Ensure initial bounds are calculated if trainMappings is already available
      if (props.trainMappings && props.trainMappings.length > 0) {
        initChart()
      }
    })

    onUnmounted(() => {
      if (chart) {
        chart.dispose()
        chart = null
      }
    })

    watch(
      () => props.classNames,
      () => initChart(),
      { deep: true },
    )
    watch(
      () => props.trainMappings,
      () => initChart(),
      { deep: true },
    )
    watch(
      () => props.testMappings,
      () => updateChart(),
      { deep: true },
    )
    watch(
      () => props.themeName,
      () => initChart(),
    )

    return {
      chartContainer,
      showBoth,
      showTrainOnly,
      showTestOnly,
      setVisibility,
      toggleAllClasses,
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

.controls-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px;
}

.legend-controls {
  display: flex;
  top: 0px;
  gap: 0.5rem;
  position: absolute;
}

.visibility-controls {
  display: flex;
  top: 0px;
  gap: 0.5rem;
  position: absolute;
  left: 50%;
  transform: translateX(-50%);
  z-index: 2;
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
