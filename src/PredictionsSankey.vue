<template>
  <div ref="containerRef" :style="{ width: '100%', height: `${diagramHeight}px` }">
    <svg ref="svgRef" :width="diagramWidth" :height="diagramHeight"></svg>
  </div>
</template>

<script lang="ts">
import { defineComponent, PropType, ref, onMounted, watch, onUnmounted } from 'vue';
import * as d3 from 'd3';
import { sankey, sankeyLinkHorizontal } from 'd3-sankey';

export default defineComponent({
  name: 'PredictionsSankey',
  props: {
    cellTypes: {
      type: Array as PropType<number[]>,
      required: true
    },
    cellTypeNames: {
      type: Array as PropType<string[]>,
      required: true
    },
    topGeneIndicesByClass: {
      type: Array as PropType<number[][]>,
      required: true
    },
    genes: {
      type: Array as PropType<string[]>,
      required: true
    }
  },
  setup(props) {
    const containerRef = ref<HTMLDivElement | null>(null);
    const svgRef = ref<SVGElement | null>(null);
    
    const diagramWidth = ref(window.innerWidth);
    const diagramHeight = ref(400);
    
    let resizeObserver: ResizeObserver | null = null;

    // Create and attach resize observer
    const setupResizeObserver = () => {
      if (!containerRef.value) return;
      
      resizeObserver = new ResizeObserver((entries) => {
        if (entries[0].contentRect) {
          diagramWidth.value = entries[0].contentRect.width;
        }
      });
      
      resizeObserver.observe(containerRef.value);
    };

    // Build and draw the sankey diagram
    const drawSankey = () => {
      if (!svgRef.value || !props.cellTypes || !props.cellTypeNames || !props.topGeneIndicesByClass || !props.genes) return;

      // Count how many cells are in each cellType
      const typeCounts: Record<string, number> = {};
      props.cellTypes.forEach((t) => {
        typeCounts[t] = (typeCounts[t] || 0) + 1;
      });

      // Subset of cell types actually present
      const presentTypes = Object.keys(typeCounts).map((key) =>
        parseInt(key, 10)
      );

      // Union of top genes across the classes found
      const foundClasses = Array.from(new Set(props.cellTypes));
      let unionGenes = new Set<number>();
      foundClasses.forEach((classIndex) => {
        const topGenes = props.topGeneIndicesByClass[classIndex] || [];
        topGenes.forEach((g) => unionGenes.add(g));
      });
      const genesArray = Array.from(unionGenes);

      // Build sankey nodes: cell types + genes
      const nodes = [
        ...presentTypes.map((t) => ({ name: props.cellTypeNames[t] || `Type${t}` })),
        ...genesArray.map((g) => ({ name: props.genes[g] })),
      ];

      // Build links: from each cell type to each top gene of that type
      const links: { source: number; target: number; value: number; }[] = [];
      for (let typeIndex = 0; typeIndex < presentTypes.length; typeIndex++) {
        // The sankey "source" is typeIndex
        const topGenes = props.topGeneIndicesByClass[presentTypes[typeIndex]] || [];
        topGenes.forEach((g) => {
          if (unionGenes.has(g)) {
            // gene is in genesArray
            const geneNodeIndex = presentTypes.length + genesArray.indexOf(g);
            links.push({
              source: typeIndex,
              target: geneNodeIndex,
              // link thickness ~ proportion of that type
              value: typeCounts[foundClasses[typeIndex]],
            });
          } else {
            console.error("Gene not found in union set:", g);
            throw new Error("Gene not found in union set");
          }
        });
      }

      // Helper functions for highlighting
      function highlight_link(id: string, opacity: number) {
        d3.select("#link-" + id).style("stroke-opacity", opacity);
      }

      function highlight_node_links(this: any, ev: Event, node: any) {
        let remainingNodes: any[] = [];
        let nextNodes: any[] = [];

        let stroke_opacity = 0;
        if (d3.select(this).attr("data-clicked") == "1") {
          d3.select(this).attr("data-clicked", "0");
          stroke_opacity = 0.5;
        } else {
          d3.select(this).attr("data-clicked", "1");
          stroke_opacity = 0.9;
        }

        const traverse = [
          {
            linkType: "sourceLinks",
            nodeType: "target",
          },
          {
            linkType: "targetLinks",
            nodeType: "source",
          },
        ];

        traverse.forEach(function (step) {
          node[step.linkType].forEach(function (link: any) {
            remainingNodes.push(link[step.nodeType]);
            highlight_link(link.id, stroke_opacity);
          });

          while (remainingNodes.length) {
            nextNodes = [];
            remainingNodes.forEach(function (node) {
              node[step.linkType].forEach(function (link: any) {
                nextNodes.push(link[step.nodeType]);
                highlight_link(link.id, stroke_opacity);
              });
            });
            remainingNodes = nextNodes;
          }
        });
      }

      // Clear previous rendering
      const svg = d3.select(svgRef.value);
      svg.selectAll("*").remove();

      // Build sankey layout
      const sankeyLayout = sankey()
        .nodeWidth(15)
        .nodePadding(10)
        .extent([
          [0, 0],
          [diagramWidth.value, diagramHeight.value],
        ]);

      const graph = sankeyLayout({
        nodes: nodes.map((d) => Object.assign({}, d)),
        links: links.map((d) => Object.assign({}, d)),
      });

      let getColor = d3.scaleOrdinal(d3.schemeCategory10);

      // Draw vertical node bars
      svg
        .append("g")
        .attr("stroke", "#000")
        .selectAll("rect")
        .data(graph.nodes)
        .join("rect")
        .attr("x", (d: any) => d.x0)
        .attr("y", (d: any) => d.y0)
        .attr("height", (d: any) => d.y1 - d.y0)
        .attr("width", (d: any) => d.x1 - d.x0)
        .attr("fill", (d: any) => getColor(d.name))
        .on("click", highlight_node_links)
        .append("title")
        .text((d: any) => `${d.name}\n${d.value}`);

      // Draw links
      const link = svg
        .append("g")
        .attr("fill", "none")
        .attr("stroke-opacity", 0.5)
        .selectAll("g")
        .data(graph.links)
        .join("g")
        .attr("id", function (d: any, i: number) {
          d.id = i;
          return "link-" + i;
        })
        .style("mix-blend-mode", "multiply");

      // Uid generator for link gradients (from Observable stdlib)
      let count = 0;
      function uid(name: string | null) {
        return new Id("O-" + (name == null ? "" : name + "-") + ++count);
      }
      function Id(id: string) {
        this.id = id;
        this.href = new URL(`#${id}`, location) + "";
      }
      Id.prototype.toString = function () {
        return "url(" + this.href + ")";
      };

      // Edge colors
      const edgeColor = "path";
      if (edgeColor === "path") {
        const gradient = link
          .append("linearGradient")
          .attr("id", (d: any) => (d.uid = uid("link")).id)
          .attr("gradientUnits", "userSpaceOnUse")
          .attr("x1", (d: any) => d.source.x1)
          .attr("x2", (d: any) => d.target.x0);

        gradient
          .append("stop")
          .attr("offset", "0%")
          .attr("stop-color", (d: any) => getColor(d.source.name));

        gradient
          .append("stop")
          .attr("offset", "100%")
          .attr("stop-color", (d: any) => getColor(d.target.name));
      }

      link
        .append("path")
        .attr("d", sankeyLinkHorizontal())
        .attr("stroke", (d: any) =>
          edgeColor === "none"
            ? "#aaa"
            : edgeColor === "path"
            ? d.uid
            : edgeColor === "input"
            ? getColor(d.source.name)
            : getColor(d.target.name)
        )
        .attr("stroke-width", (d: any) => Math.max(1, d.width));

      link
        .append("title")
        .text((d: any) => `${d.source.name} → ${d.target.name}\n${d.value}`);

      svg
        .append("g")
        .attr("font-family", "sans-serif")
        .attr("font-size", 10)
        .selectAll("text")
        .data(graph.nodes)
        .join("text")
        .attr("x", (d: any) => (d.x0 < diagramWidth.value / 2 ? d.x1 + 6 : d.x0 - 6))
        .attr("y", (d: any) => (d.y1 + d.y0) / 2)
        .attr("dy", "0.35em")
        .attr("text-anchor", (d: any) => (d.x0 < diagramWidth.value / 2 ? "start" : "end"))
        .text((d: any) => d.name);
    };

    // Draw the diagram when component is mounted
    onMounted(() => {
      setupResizeObserver();
      drawSankey();
    });

    // Redraw when props or size changes
    watch(
      [
        () => props.cellTypes,
        () => props.cellTypeNames,
        () => props.topGeneIndicesByClass,
        () => props.genes,
        diagramWidth,
        diagramHeight
      ],
      () => {
        drawSankey();
      }
    );

    // Clean up observer when component is unmounted
    onUnmounted(() => {
      if (resizeObserver) {
        resizeObserver.disconnect();
      }
    });

    return {
      containerRef,
      svgRef,
      diagramWidth,
      diagramHeight
    };
  }
});
</script>

<style scoped>
/* Component-specific styles if needed */
</style>
