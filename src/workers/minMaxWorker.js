/**
 * Analytics worker that computes the min/max of a batch
 */

// Handle messages from the main thread
self.addEventListener('message', async (event) => {
  const { type, batch, batchSize, numGenes } = event.data
  
  if (type === 'computeMinMax') {
    try {
      self.postMessage({ type: 'status', message: 'Computing min/max...' })
      
      // In a real implementation, this worker would load an ONNX model
      // For now, just compute the min/max of the batch
      const min = new Float32Array(numGenes)
      const max = new Float32Array(numGenes)
      
      // Initialize min to maximum possible value and max to minimum possible value
      for (let i = 0; i < numGenes; i++) {
        min[i] = Number.MAX_VALUE
        max[i] = Number.MIN_VALUE
      }
      
      // Compute min/max per gene across all cells in the batch
      for (let geneIdx = 0; geneIdx < numGenes; geneIdx++) {
        for (let cellIdx = 0; cellIdx < batchSize; cellIdx++) {
          const value = batch[cellIdx * numGenes + geneIdx]
          min[geneIdx] = Math.min(min[geneIdx], value)
          max[geneIdx] = Math.max(max[geneIdx], value)
        }
      }
      
      // Find global min/max for summary
      let globalMin = Number.MAX_VALUE
      let globalMax = Number.MIN_VALUE
      
      for (let i = 0; i < numGenes; i++) {
        globalMin = Math.min(globalMin, min[i])
        globalMax = Math.max(globalMax, max[i])
      }
      
      self.postMessage({
        type: 'analysisResult',
        result: {
          type: 'minMax',
          min,
          max,
          summary: `Min: ${globalMin.toFixed(6)}, Max: ${globalMax.toFixed(6)}`
        },
        message: `Min/max computation complete: Min=${globalMin.toFixed(6)}, Max=${globalMax.toFixed(6)}`
      })
      
    } catch (error) {
      self.postMessage({ 
        type: 'error', 
        message: `Error in min/max worker: ${error.message}`
      })
    }
  }
})
