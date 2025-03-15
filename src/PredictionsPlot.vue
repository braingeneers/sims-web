<template>
  <svg :width="width" :height="height" id="predictions-plot" ref="plotRef"></svg>
</template>

<script lang="ts">
import { defineComponent, PropType, ref, onMounted, watch } from 'vue'
import * as d3 from 'd3'

export default defineComponent({
  name: 'PredictionsPlot',
  props: {
    width: {
      type: Number,
      required: true,
    },
    height: {
      type: Number,
      required: true,
    },
    labels: {
      type: Array as PropType<number[]>,
      required: true,
    },
    coordinates: {
      type: Array as PropType<number[]>,
      required: true,
    },
  },
  setup(props) {
    const plotRef = ref<SVGElement | null>(null)

    // Function to draw the plot
    const drawPlot = () => {
      if (!plotRef.value) return

      // Clear previous content
      d3.select(plotRef.value).selectAll('*').remove()

      const margin = { top: 10, right: 30, bottom: 30, left: 40 }
      const inner_width = props.width - margin.left - margin.right
      const inner_height = props.height - margin.top - margin.bottom

      const svg = d3
        .select(plotRef.value)
        .append('svg')
        .attr('width', inner_width + margin.left + margin.right)
        .attr('height', inner_height + margin.top + margin.bottom)
        .append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`)

      // Convert 1d array of [x,y,x,y..] coordinates to 1d array of [x, y] pairs
      function* coordIterator(coords: number[]) {
        for (let i = 0; i < coords.length; i += 2) {
          yield [coords[i], coords[i + 1]]
        }
      }

      const x = d3
        .scaleLinear()
        .domain(
          d3.extent(
            Array.from(coordIterator(props.coordinates)),
            (d: [number, number]) => d[0],
          ) as [number, number],
        )
        .range([0, inner_width])

      const y = d3
        .scaleLinear()
        .domain(
          d3.extent(
            Array.from(coordIterator(props.coordinates)),
            (d: [number, number]) => d[1],
          ) as [number, number],
        )
        .range([inner_height, 0])

      svg.append('g').attr('transform', `translate(0,${inner_height})`).call(d3.axisBottom(x))

      svg.append('g').call(d3.axisLeft(y))

      const color = d3.scaleOrdinal(d3.schemeCategory10)

      svg
        .selectAll('circle')
        .data(Array.from(coordIterator(props.coordinates)))
        .enter()
        .append('circle')
        .attr('cx', (d: [number, number]) => x(d[0]))
        .attr('cy', (d: [number, number]) => y(d[1]))
        .attr('r', 3)
        .style('fill', (_d: [number, number], i: number) => color(props.labels[i]))
    }

    // Draw the plot when component is mounted
    onMounted(() => {
      drawPlot()
    })

    // Redraw when props change
    watch(
      [() => props.coordinates, () => props.labels, () => props.width, () => props.height],
      () => {
        drawPlot()
      },
    )

    return {
      plotRef,
    }
  },
})
</script>

<style scoped>
/* Component-specific styles if needed */
</style>
