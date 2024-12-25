import * as d3 from "d3";
import { useEffect, useRef } from "react";
import PropTypes from "prop-types";

export const PredictionsPlot = ({ width, height, predictions }) => {
  const ref = useRef();

  useEffect(() => {
    // set the dimensions and margins of the graph
    d3.select(ref.current).selectAll("*").remove();

    if (!predictions) {
      return () => {}; // Return an empty function
    }

    const margin = { top: 10, right: 30, bottom: 30, left: 40 };
    const inner_width = width - margin.left - margin.right;
    const inner_height = height - margin.top - margin.bottom;

    const svg = d3
      .select(ref.current)
      .append("svg")
      .attr("width", inner_width + margin.left + margin.right)
      .attr("height", inner_height + margin.top + margin.bottom)
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    const x = d3
      .scaleLinear()
      .domain(d3.extent(predictions.coordinates, (d) => d[0]))
      .range([0, inner_width]);
    const y = d3
      .scaleLinear()
      .domain(d3.extent(predictions.coordinates, (d) => d[1]))
      .range([inner_height, 0]);
    svg
      .append("g")
      .attr("transform", `translate(0,${inner_height})`)
      .call(d3.axisBottom(x));
    svg.append("g").call(d3.axisLeft(y));

    const color = d3.scaleOrdinal(d3.schemeCategory10);
    svg
      .selectAll("circle")
      .data(predictions.coordinates)
      .enter()
      .append("circle")
      .attr("cx", (d) => x(d[0]))
      .attr("cy", (d) => y(d[1]))
      .attr("r", 3)
      .style("fill", (d, i) => color(predictions.labels[i][0][0]));
  });

  return <svg width={width} height={height} id="predictions-plot" ref={ref} />;
};

PredictionsPlot.propTypes = {
  width: PropTypes.number,
  height: PropTypes.number,
  predictions: PropTypes.shape({
    labels: PropTypes.arrayOf(
      PropTypes.arrayOf([PropTypes.string, PropTypes.float])
    ),
    coordinates: PropTypes.arrayOf(
      PropTypes.arrayOf([PropTypes.float, PropTypes.float])
    ),
  }),
};
