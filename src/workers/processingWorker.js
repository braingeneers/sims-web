/**
 * Second worker in the pipeline: Receives batch data from the file worker
 * Just passes the data to the next worker in this implementation
 */

// Handle messages from the main thread
self.addEventListener('message', async (event) => {
  const { type, batch, batchSize, numCells, numGenes } = event.data
  
  if (type === 'processBatch') {
    try {
      self.postMessage({ type: 'status', message: 'Processing batch...' })
      
      // In a real implementation, this worker would load an ONNX model
      // and process the batch data using the model
      
      // For now, just pass the data to the next worker
      self.postMessage({
        type: 'batchProcessed',
        batch,
        batchSize,
        numCells,
        numGenes,
        message: 'Batch forwarded to next worker'
      })
      
    } catch (error) {
      self.postMessage({ 
        type: 'error', 
        message: `Error in processing worker: ${error.message}`
      })
    }
  }
})
