import PropTypes from "prop-types";
import Box from "@mui/material/Box";
import Table from "@mui/material/Table";
import TableBody from "@mui/material/TableBody";
import TableCell from "@mui/material/TableCell";
import TableContainer from "@mui/material/TableContainer";
import TableHead from "@mui/material/TableHead";
import TableRow from "@mui/material/TableRow";
import Typography from "@mui/material/Typography";
import IconButton from "@mui/material/IconButton";
import DownloadIcon from "@mui/icons-material/Download";

export const PredictionsTable = ({
  cellNames,
  predictions,
  probabilities,
  cellTypeNames,
  coordinates,
}) => {
  function downloadCSV() {
    let csvContent =
      "cell_id,pred_0,pred_1,pred_2,prob_0,prob_1,prob_2,umap_0,umap_1\n";

    cellNames.forEach((cellName, cellIndex) => {
      let cellResults = "";
      for (let i = 0; i < 3; i++) {
        cellResults += `,${cellTypeNames[predictions[cellIndex][i]]}`;
      }
      for (let i = 0; i < 3; i++) {
        cellResults += `,${probabilities[cellIndex][i].toFixed(4)}`;
      }
      if (cellIndex < coordinates.length / 2) {
        cellResults += `,${coordinates[2 * cellIndex].toFixed(4)},${coordinates[
          2 * cellIndex + 1
        ].toFixed(4)}`;
      } else {
        // If there are no UMAP coordinates, just add a comma
        cellResults += ",,";
      }
      csvContent += `${cellName}${cellResults}\n`;
    });

    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const downloadLink = document.createElement("a");
    const url = URL.createObjectURL(blob);
    downloadLink.setAttribute("href", url);
    downloadLink.setAttribute("download", "predictions.csv");
    document.body.appendChild(downloadLink);
    downloadLink.click();
    document.body.removeChild(downloadLink);
  }

  return (
    <TableContainer>
      <Box mt={2}>
        <Typography>First 100 predictions:</Typography>
        <IconButton
          onClick={() => downloadCSV(predictions)}
          disabled={predictions === null}
          color="primary"
          style={{ float: "right" }}
        >
          <DownloadIcon />
        </IconButton>
      </Box>
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
          {cellNames.slice(0, 100).map((cellName, cellIndex) => (
            <TableRow
              key={cellName}
              sx={{ "&:last-child td, &:last-child th": { border: 0 } }}
            >
              <TableCell component="th" scope="row">
                {cellName}
              </TableCell>
              <TableCell>{cellTypeNames[predictions[cellIndex][0]]}</TableCell>
              <TableCell>{probabilities[cellIndex][0].toFixed(4)}</TableCell>
              <TableCell>{cellTypeNames[predictions[cellIndex][1]]}</TableCell>
              <TableCell>{probabilities[cellIndex][1].toFixed(4)}</TableCell>
              <TableCell>{cellTypeNames[predictions[cellIndex][2]]}</TableCell>
              <TableCell>{probabilities[cellIndex][2].toFixed(4)}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );
};

PredictionsTable.propTypes = {
  cellNames: PropTypes.arrayOf(PropTypes.string),
  predictions: PropTypes.arrayOf(PropTypes.arrayOf(PropTypes.number)),
  probabilities: PropTypes.arrayOf(PropTypes.instanceOf(Float32Array)),
  cellTypeNames: PropTypes.arrayOf(PropTypes.string),
  coordinates: PropTypes.arrayOf(PropTypes.number),
};
