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

export function PredictionsSankey({ datasetLabel, width = 600, height = 400 }) {
  const svgRef = useRef(null);
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

    // Clear previous render
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    // Build sankey layout
    const sankeyLayout = sankey()
      .nodeWidth(15)
      .nodePadding(10)
      .extent([
        [0, 0],
        [width, height],
      ]);

    const graph = sankeyLayout({
      nodes: nodes.map((d) => Object.assign({}, d)),
      links: links.map((d) => Object.assign({}, d)),
    });

    // Draw links
    svg
      .append("g")
      .selectAll("path")
      .data(graph.links)
      .join("path")
      .attr("d", sankeyLinkHorizontal())
      .attr("fill", "none")
      .attr("stroke", "#aaa")
      .attr("stroke-width", (d) => d.width)
      .attr("opacity", 0.7);

    // Draw nodes
    const node = svg
      .append("g")
      .selectAll("g")
      .data(graph.nodes)
      .join("g")
      .attr("transform", (d) => `translate(${d.x0},${d.y0})`);

    node
      .append("rect")
      .attr("x", 0)
      .attr("y", 0)
      .attr("height", (d) => d.y1 - d.y0)
      .attr("width", (d) => d.x1 - d.x0)
      .attr("fill", "#69b3a2");

    node
      .append("text")
      .attr("x", -6)
      .attr("y", (d) => (d.y1 - d.y0) / 2)
      .attr("dy", "0.35em")
      .attr("text-anchor", "end")
      .text((d) => d.name)
      .filter((d) => d.x0 < width / 2)
      .attr("x", 6 + (graph.nodes[0]?.x1 - graph.nodes[0]?.x0 || 0))
      .attr("text-anchor", "start");
  }, [dbData, width, height]);

  return <svg ref={svgRef} width={width} height={height}></svg>;
}
