import Table from "@mui/material/Table";
import TableBody from "@mui/material/TableBody";
import TableCell from "@mui/material/TableCell";
import TableContainer from "@mui/material/TableContainer";
import TableHead from "@mui/material/TableHead";
import TableRow from "@mui/material/TableRow";

export function downloadCSV(predictions) {
  if (!predictions) {
    return;
  }
  let csvContent =
    "cell_id,p0_class,p0_prob,p1_class,p1_prob,p2_class,p2_prob,umap_0,umap_1\n";

  predictions.cellNames.forEach((cellName, cellIndex) => {
    let cellResults = "";
    for (let i = 0; i < 3; i++) {
      cellResults += `,${
        predictions.classes[predictions.labels[cellIndex][0][i]]
      },${predictions.labels[cellIndex][1][i].toFixed(4)}`;
    }
    cellResults += `,${predictions.coordinates[cellIndex][0].toFixed(
      4
    )},${predictions.coordinates[cellIndex][1].toFixed(4)}`;
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

export function PredictionsTable({ predictions }) {
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
}
