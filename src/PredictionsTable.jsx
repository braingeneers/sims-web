// import React from "react";
import PropTypes from "prop-types";
import Table from "@mui/material/Table";
import TableBody from "@mui/material/TableBody";
import TableCell from "@mui/material/TableCell";
import TableContainer from "@mui/material/TableContainer";
import TableHead from "@mui/material/TableHead";
import TableRow from "@mui/material/TableRow";

export const PredictionsTable = ({ predictions }) => {
  if (!predictions) {
    return <div></div>; // Return an empty div
  }
  return (
    <TableContainer>
      <Table sx={{ minWidth: 650 }} aria-label="simple table">
        <TableHead>
          <TableRow>
            <TableCell>Cell</TableCell>
            <TableCell>Class 1</TableCell>
            <TableCell>Prob 1</TableCell>
            <TableCell>Class 2</TableCell>
            <TableCell>Prob 2</TableCell>
            <TableCell>Class 3</TableCell>
            <TableCell>Prob 3</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {predictions.cellNames.map((cellName, cellIndex) => (
            <TableRow
              key={cellName}
              sx={{ "&:last-child td, &:last-child th": { border: 0 } }}
            >
              <TableCell component="th" scope="row">
                {cellName}
              </TableCell>
              <TableCell>
                {predictions.classes[predictions.labels[cellIndex][0][0]]}
              </TableCell>
              <TableCell>
                {predictions.labels[cellIndex][1][0].toFixed(4)}
              </TableCell>
              <TableCell>
                {predictions.classes[predictions.labels[cellIndex][0][1]]}
              </TableCell>
              <TableCell>
                {predictions.labels[cellIndex][1][1].toFixed(4)}
              </TableCell>
              <TableCell>
                {predictions.classes[predictions.labels[cellIndex][0][2]]}
              </TableCell>
              <TableCell>
                {predictions.labels[cellIndex][1][2].toFixed(4)}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );
};

PredictionsTable.propTypes = {
  predictions: PropTypes.shape({
    labels: PropTypes.arrayOf(
      PropTypes.arrayOf([PropTypes.string, PropTypes.float])
    ),
    coordinates: PropTypes.arrayOf(
      PropTypes.arrayOf([PropTypes.float, PropTypes.float])
    ),
    cellNames: PropTypes.arrayOf(PropTypes.string),
    classes: PropTypes.arrayOf(PropTypes.string),
  }),
};
