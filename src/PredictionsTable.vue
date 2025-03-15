<template>
  <div>
    <div class="mt-2">
      <p>First 100 predictions:</p>
      <v-btn
        icon
        @click="downloadCSV"
        :disabled="!predictions"
        color="primary"
        style="float: right"
      >
        <v-icon>mdi-download</v-icon>
      </v-btn>
    </div>
    <v-table>
      <thead>
        <tr>
          <th>Cell</th>
          <th>Class 1</th>
          <th>Prob 1</th>
          <th>Class 2</th>
          <th>Prob 2</th>
          <th>Class 3</th>
          <th>Prob 3</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="(cellName, cellIndex) in cellNames.slice(0, 100)" :key="cellName">
          <td>{{ cellName }}</td>
          <td>{{ cellTypeNames[predictions[cellIndex][0]] }}</td>
          <td>{{ probabilities[cellIndex][0].toFixed(4) }}</td>
          <td>{{ cellTypeNames[predictions[cellIndex][1]] }}</td>
          <td>{{ probabilities[cellIndex][1].toFixed(4) }}</td>
          <td>{{ cellTypeNames[predictions[cellIndex][2]] }}</td>
          <td>{{ probabilities[cellIndex][2].toFixed(4) }}</td>
        </tr>
      </tbody>
    </v-table>
  </div>
</template>

<script lang="ts">
import { defineComponent, PropType } from 'vue';

export default defineComponent({
  name: 'PredictionsTable',
  props: {
    cellNames: {
      type: Array as PropType<string[]>,
      required: true
    },
    predictions: {
      type: Array as PropType<number[][]>,
      required: true
    },
    probabilities: {
      type: Array as PropType<Float32Array[]>,
      required: true
    },
    cellTypeNames: {
      type: Array as PropType<string[]>,
      required: true
    },
    coordinates: {
      type: Array as PropType<number[]>,
      required: true
    }
  },
  setup(props) {
    const downloadCSV = () => {
      let csvContent =
        "cell_id,pred_0,pred_1,pred_2,prob_0,prob_1,prob_2,umap_0,umap_1\n";

      props.cellNames.forEach((cellName, cellIndex) => {
        let cellResults = "";
        for (let i = 0; i < 3; i++) {
          cellResults += `,${props.cellTypeNames[props.predictions[cellIndex][i]]}`;
        }
        for (let i = 0; i < 3; i++) {
          cellResults += `,${props.probabilities[cellIndex][i].toFixed(4)}`;
        }
        if (cellIndex < props.coordinates.length / 2) {
          cellResults += `,${props.coordinates[2 * cellIndex].toFixed(4)},${props.coordinates[
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
    };

    return {
      downloadCSV
    };
  }
});
</script>

<style scoped>
/* Component-specific styles if needed */
</style>
