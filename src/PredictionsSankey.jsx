/**
 * SankeyPlot component
 * - Fetches the specified dataset from the "sims-web" IndexedDB
 * - Builds a Sankey diagram showing:
 *   Left: cell types found (thickness ~ proportion)
 *   Middle: low and high confidence prediction
 *   Right: union of all top gene indices (thickness ~ link to cell type)
 *
 * See:
 * https://observablehq.com/@iashishsingh/sankey-diagram-path-highlighting
 */
import React, { useEffect, useRef, useState } from "react";
import * as d3 from "d3";
import { sankey, sankeyLinkHorizontal } from "d3-sankey";

export function PredictionsSankey({ datasetLabel }) {
  // Reference to our container DIV
  const containerRef = useRef(null);
  // Reference to our SVG
  const svgRef = useRef(null);

  // Track window width in state
  const [diagramWidth, setDiagramWidth] = useState(window.innerWidth);
  const [diagramHeight, setDiagramHeight] = useState(400);

  // Observe container size changes
  useEffect(() => {
    if (!containerRef.current) return;

    const resizeObserver = new ResizeObserver((entries) => {
      if (entries[0].contentRect) {
        setDiagramWidth(entries[0].contentRect.width);
      }
    });
    resizeObserver.observe(containerRef.current);

    return () => resizeObserver.disconnect();
  }, []);

  const [dbData, setDbData] = useState(null);

  // Load data from IndexedDB
  useEffect(() => {
    const request = indexedDB.open("sims-web", 1);
    request.onsuccess = () => {
      const db = request.result;
      const tx = db.transaction("datasets", "readonly");
      const store = tx.objectStore("datasets");
      const getRequest = store.get(datasetLabel); // fetch by datasetLabel
      getRequest.onsuccess = () => {
        const result = getRequest.result;
        if (result) {
          setDbData(result);
        }
        db.close();
      };
    };
  }, [datasetLabel]);

  // Build and draw sankey when dbData changes
  useEffect(() => {
    if (!dbData || !svgRef.current) return;

    const { cellTypes, cellTypeNames, topGeneIndicesByClass, genes } = dbData;
    if (!cellTypes || !cellTypeNames || !topGeneIndicesByClass || !genes)
      return;

    // Count how many cells are in each cellType
    const typeCounts = {};
    cellTypes.forEach((t) => {
      typeCounts[t] = (typeCounts[t] || 0) + 1;
    });

    // Subset of cell types actually present
    const presentTypes = Object.keys(typeCounts).map((key) =>
      parseInt(key, 10)
    );

    // Union of top genes across the classes found
    const foundClasses = Array.from(new Set(cellTypes));
    let unionGenes = new Set();
    foundClasses.forEach((classIndex) => {
      const topGenes = topGeneIndicesByClass[classIndex] || [];
      topGenes.forEach((g) => unionGenes.add(g));
    });
    const genesArray = Array.from(unionGenes);

    // Build sankey nodes: cell types + genes
    // Node index 0..(presentTypes.length - 1) for cell types
    // Then continuing for gene nodes
    const nodes = [
      ...presentTypes.map((t) => ({ name: cellTypeNames[t] || `Type${t}` })),
      ...genesArray.map((g) => ({ name: genes[g] })),
    ];

    // Build links: from each cell type to each top gene of that type
    // Use the class index to find topGeneIndices, link flows to each gene
    const links = [];
    for (let typeIndex = 0; typeIndex < presentTypes.length; typeIndex++) {
      // The sankey "source" is typeIndex
      const topGenes = topGeneIndicesByClass[presentTypes[typeIndex]] || [];
      topGenes.forEach((g) => {
        if (unionGenes.has(g)) {
          // gene is in genesArray
          const geneNodeIndex = presentTypes.length + genesArray.indexOf(g);
          links.push({
            source: typeIndex,
            target: geneNodeIndex,
            // link thickness ~ proportion of that type
            value: typeCounts[foundClasses[typeIndex]],
            // value: typeCounts[typeIndex] / cellTypes.length,
          });
        } else {
          console.error("Gene not found in union set:", g);
          throw new Error("Gene not found in union set");
        }
      });
    }

    // ===========================================================
    // Hightlighing nodes and links
    // ===========================================================
    function highlight_node_links(ev, node) {
      var remainingNodes = [],
        nextNodes = [];

      var stroke_opacity = 0;
      if (d3.select(this).attr("data-clicked") == "1") {
        d3.select(this).attr("data-clicked", "0");
        stroke_opacity = 0.5;
      } else {
        d3.select(this).attr("data-clicked", "1");
        stroke_opacity = 0.9;
      }

      var traverse = [
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
        node[step.linkType].forEach(function (link) {
          remainingNodes.push(link[step.nodeType]);
          highlight_link(link.id, stroke_opacity);
        });

        while (remainingNodes.length) {
          nextNodes = [];
          remainingNodes.forEach(function (node) {
            node[step.linkType].forEach(function (link) {
              nextNodes.push(link[step.nodeType]);
              highlight_link(link.id, stroke_opacity);
            });
          });
          remainingNodes = nextNodes;
        }
      });
    }

    function highlight_link(id, opacity) {
      d3.select("#link-" + id).style("stroke-opacity", opacity);
    }

    // Clear previous render
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    // ===========================================================
    // Build sankey layout
    // ===========================================================
    const sankeyLayout = sankey()
      .nodeWidth(15)
      .nodePadding(10)
      .extent([
        [0, 0],
        [diagramWidth, diagramHeight],
      ]);

    const graph = sankeyLayout({
      nodes: nodes.map((d) => Object.assign({}, d)),
      links: links.map((d) => Object.assign({}, d)),
    });

    // ===========================================================
    // New sankey diagram
    // ===========================================================

    let getColor = d3.scaleOrdinal(d3.schemeCategory10);

    // Draw vertical node bars
    svg
      .append("g")
      .attr("stroke", "#000")
      .selectAll("rect")
      .data(graph.nodes)
      .join("rect")
      .attr("x", (d) => d.x0)
      .attr("y", (d) => d.y0)
      .attr("height", (d) => d.y1 - d.y0)
      .attr("width", (d) => d.x1 - d.x0)
      .attr("fill", (d) => getColor(d.name))
      .on("click", highlight_node_links)
      .append("title")
      .text((d) => `${d.name}\n${d.value}`);

    // Draw links
    const link = svg
      .append("g")
      .attr("fill", "none")
      .attr("stroke-opacity", 0.5)
      .selectAll("g")
      .data(graph.links)
      .join("g")
      .attr("id", function (d, i) {
        d.id = i;
        return "link-" + i;
      })
      .style("mix-blend-mode", "multiply");

    // Code from observable library so we can run outside observable
    // https://github.com/observablehq/stdlib/blob/main/src/dom/uid.js
    var count = 0;
    function uid(name) {
      return new Id("O-" + (name == null ? "" : name + "-") + ++count);
    }
    function Id(id) {
      this.id = id;
      this.href = new URL(`#${id}`, location) + "";
    }
    Id.prototype.toString = function () {
      return "url(" + this.href + ")";
    };

    const edgeColor = "path";
    if (edgeColor === "path") {
      const gradient = link
        .append("linearGradient")
        .attr("id", (d) => (d.uid = uid("link")).id)
        .attr("gradientUnits", "userSpaceOnUse")
        .attr("x1", (d) => d.source.x1)
        .attr("x2", (d) => d.target.x0);

      gradient
        .append("stop")
        .attr("offset", "0%")
        .attr("stop-color", (d) => getColor(d.source));

      gradient
        .append("stop")
        .attr("offset", "100%")
        .attr("stop-color", (d) => getColor(d.target));
    }

    link
      .append("path")
      .attr("d", sankeyLinkHorizontal())
      .attr("stroke", (d) =>
        edgeColor === "none"
          ? "#aaa"
          : edgeColor === "path"
          ? d.uid
          : edgeColor === "input"
          ? getColor(d.source)
          : getColor(d.target)
      )
      .attr("stroke-width", (d) => Math.max(1, d.width));

    link
      .append("title")
      .text((d) => `${d.source.name} → ${d.target.name}\n${d.value}`);

    svg
      .append("g")
      .attr("font-family", "sans-serif")
      .attr("font-size", 10)
      .selectAll("text")
      .data(graph.nodes)
      .join("text")
      .attr("x", (d) => (d.x0 < diagramWidth / 2 ? d.x1 + 6 : d.x0 - 6))
      .attr("y", (d) => (d.y1 + d.y0) / 2)
      .attr("dy", "0.35em")
      .attr("text-anchor", (d) => (d.x0 < diagramWidth / 2 ? "start" : "end"))
      .text((d) => d.name);

    // ===========================================================
    // Working sankey diagram
    // ===========================================================

    // // Draw links
    // svg
    //   .append("g")
    //   .selectAll("path")
    //   .data(graph.links)
    //   .join("path")
    //   .attr("d", sankeyLinkHorizontal())
    //   .attr("fill", "none")
    //   .attr("stroke", "#aaa")
    //   .attr("stroke-width", (d) => d.width)
    //   .attr("opacity", 0.7);

    // // Draw nodes
    // const node = svg
    //   .append("g")
    //   .selectAll("g")
    //   .data(graph.nodes)
    //   .join("g")
    //   .attr("transform", (d) => `translate(${d.x0},${d.y0})`);

    // node
    //   .append("rect")
    //   .attr("x", 0)
    //   .attr("y", 0)
    //   .attr("height", (d) => d.y1 - d.y0)
    //   .attr("width", (d) => d.x1 - d.x0)
    //   .attr("fill", "#69b3a2");

    // node
    //   .append("text")
    //   .attr("x", -6)
    //   .attr("y", (d) => (d.y1 - d.y0) / 2)
    //   .attr("dy", "0.35em")
    //   .attr("text-anchor", "end")
    //   .text((d) => d.name)
    //   .filter((d) => d.x0 < diagramWidth / 2)
    //   .attr("x", 6 + (graph.nodes[0]?.x1 - graph.nodes[0]?.x0 || 0))
    //   .attr("text-anchor", "start");
  }, [dbData, diagramWidth, diagramHeight]);

  return (
    <div ref={containerRef} style={{ width: "100%", height: diagramHeight }}>
      <svg ref={svgRef} width={diagramWidth} height={diagramHeight}></svg>
    </div>
  );
}
