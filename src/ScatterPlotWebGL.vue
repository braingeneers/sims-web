<template>
  <div ref="chartContainer" style="width: 100%; height: 400px"></div>
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
        pieces: pieces, // Use named pieces
        outOfRange: {
          symbolSize: 0,
        },
        show: true,
        orient: 'vertical', // Optional styling
        left: 10,
        top: 'center',
      },
      series: [
        {
          name: 'Reference',
          type: 'scatterGL', // WebGL renderer
          data: props.trainMappings, // Use props.trainMappings instead of data
          dimensions: ['x', 'y', 'class'],
          symbolSize: 1,
          itemStyle: {
            opacity: 0.07,
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
      () => {
        if (chart) {
          const updatedOption = {
            series: [{
              data: props.trainMappings
            }, {
              data: props.testMappings
            }]
          }
          chart.setOption(updatedOption, { notMerge: false })
        }
      },
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
    }
  },
})
</script>

<style scoped>
/* Ensure the chart container takes full width and height */
div {
  width: 100%;
  height: 100%;
}
</style>
