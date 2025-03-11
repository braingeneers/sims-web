/**
 * Analytics worker that computes the average of a batch
 */

// Handle messages from the main thread
self.addEventListener('message', async (event) => {
  const { type, batch, batchSize, numGenes } = event.data
  
  if (type === 'computeAverage') {
    try {
      self.postMessage({ type: 'status', message: 'Computing average...' })
      
      // In a real implementation, this worker would load an ONNX model
      // For now, just compute the average of the batch
      const average = new Float32Array(numGenes)
      
      // Compute average per gene across all cells in the batch
      for (let geneIdx = 0; geneIdx < numGenes; geneIdx++) {
        let sum = 0
        for (let cellIdx = 0; cellIdx < batchSize; cellIdx++) {
          sum += batch[cellIdx * numGenes + geneIdx]
        }
        average[geneIdx] = sum / batchSize
      }
      
      // Compute overall average for summary
      let overallAvg = 0
      for (let i = 0; i < average.length; i++) {
        overallAvg += average[i]
      }
      overallAvg /= average.length
      
      self.postMessage({
        type: 'analysisResult',
        result: {
          type: 'average',
          detailedResult: average,
          summary: overallAvg.toFixed(6)
        },
        message: `Average computation complete: ${overallAvg.toFixed(6)}`
      })
      
    } catch (error) {
      self.postMessage({ 
        type: 'error', 
        message: `Error in average worker: ${error.message}`
      })
    }
  }
})
