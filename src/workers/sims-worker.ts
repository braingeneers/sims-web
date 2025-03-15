/**
 * Browser Web Worker that runs an ONNX model on an h5ad file and stores the
 * results in IndexedDB. These results include:
 * - Cell type predictions with probabilities
 * - Ranked genes per class that explain the prediction (based on attention)
 * - UMAP coordinates for each cell
 */
import h5wasm from 'h5wasm'
import Prando from 'prando'
import * as UMAP from 'umap-js'
import * as ort from 'onnxruntime-web'
import { openDB } from 'idb'

// Define interfaces for TypeScript
interface WorkerModel {
  modelID: string
  session: ort.InferenceSession
  genes: string[]
  classes: string[]
}

interface PredictionResult {
  datasetLabel: string
  coords: Float32Array
  cellTypeNames: string[]
  cellTypes: number[]
  modelID: string
  topGeneIndicesByClass: number[][]
  genes: string[]
  overallTopGenes: number[]
  cellNames: string[]
  predictions: number[][]
  probabilities: Float32Array[]
}

// Extend the self object for TypeScript
declare global {
  interface Worker {
    model: WorkerModel | null
    numExplainGenes: number
    attentionAccumulators: number[][] | null
    numThreads: number
    batchSize: number
    maxNumCellsToUMAP: number
  }
}

// Dictionary with various model information (id, genes, session)
self.model = null

// Top number of genes to return to explain per class (i.e. Top K)
self.numExplainGenes = 10

// #classes x #genes matrix to accumulate attention for explaining predictions
self.attentionAccumulators = null

// Number of threads to use for inference. Use all but one for the GUI to run in
self.numThreads = navigator.hardwareConcurrency - 1

// Batch size based on available threads
self.batchSize = self.numThreads - 1

console.log(`Number of threads: ${self.numThreads}`)
console.log(`Batch size: ${self.batchSize}`)

// Limit how many UMAP points we calculate
self.maxNumCellsToUMAP = 2000

// Handle messages from the main thread
self.addEventListener('message', async function (event: MessageEvent) {
  if (event.data.type === 'startPrediction') {
    predict(event.data.modelID, event.data.modelURL, event.data.h5File, event.data.cellRangePercent)
  }
})

/**
 * Create an ONNX Runtime session for the selected model
 */
async function instantiateModel(modelURL: string, modelID: string): Promise<WorkerModel> {
  console.log(`Instantiating model ${modelID} from ${modelURL}`)
  self.postMessage({ type: 'status', message: 'Downloading model...' })

  // Fetch the model gene list
  let response = await fetch(`${modelURL}/${modelID}.genes`)
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`)
  }
  const genes = (await response.text()).split('\n')
  console.log('Model Genes', genes.slice(0, 5))

  // Fetch the model classes
  response = await fetch(`${modelURL}/${modelID}.classes`)
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`)
  }
  const classes = (await response.text()).split('\n')
  console.log('Model Classes', classes)

  // Fetch the model ONNX file incrementally to show progress
  response = await fetch(`${modelURL}/${modelID}.onnx`)
  if (!response.ok) {
    throw new Error(`Error fetching onnx file: ${response.status}`)
  }
  const contentLength = response.headers.get('content-length')
  if (!contentLength) {
    throw new Error('Content-Length header is missing')
  }
  const totalBytes = parseInt(contentLength, 10)
  let loadedBytes = 0

  // Read the response body as a stream
  const reader = response.body!.getReader()
  const chunks: Uint8Array[] = []

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    chunks.push(value)
    loadedBytes += value.length

    // Send progress update to the main thread
    self.postMessage({
      type: 'predictionProgress',
      message: 'Downloading model...',
      countFinished: loadedBytes,
      totalToProcess: totalBytes,
    })
  }

  // Combine all chunks into a single ArrayBuffer
  const modelArray = new Uint8Array(loadedBytes)
  let position = 0
  for (const chunk of chunks) {
    modelArray.set(chunk, position)
    position += chunk.length
  }

  // Initialize ONNX Runtime environment
  self.postMessage({ type: 'status', message: 'Instantiating model...' })
  // See https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html
  ort.env.wasm.numThreads = self.numThreads
  ort.env.wasm.proxy = true
  ort.env.debug = true
  ort.env.logLevel = 'verbose'
  const options: ort.InferenceSession.SessionOptions = {
    executionProviders: ['wasm'],
    executionMode: 'parallel',
  }

  // Create the InferenceSession with the model ArrayBuffer
  const session = await ort.InferenceSession.create(modelArray.buffer, options)
  console.log('Model Output names', session.outputNames)

  return { modelID, session, genes, classes }
}

/**
 * Precompute the inflation indices for the sample gene list
 */
function precomputeInflationIndices(currentModelGenes: string[], sampleGenes: string[]): number[] {
  const inflationIndices: number[] = []
  for (let geneIndex = 0; geneIndex < sampleGenes.length; geneIndex++) {
    inflationIndices.push(currentModelGenes.indexOf(sampleGenes[geneIndex]))
  }
  const missingGenesInModel = inflationIndices.filter((x) => x === -1).length
  console.log(`Missing genes in model: ${missingGenesInModel}`)
  return inflationIndices
}

/**
 * Fill the batch data and inflate it into the model's gene list space
 */
function fillBatchData(
  batchStart: number,
  currentBatchSize: number,
  data: Float32Array | Float64Array,
  indices: Uint32Array | Int32Array | null,
  indptr: Uint32Array | Int32Array | null,
  isSparse: boolean,
  sampleGenes: string[],
  inflationIndices: number[],
  encodings: Float32Array,
  numGenes: number,
): void {
  for (let i = 0; i < currentBatchSize; i++) {
    const cellIndex = batchStart + i
    const batchOffset = i * numGenes

    // Initialize batch data
    for (let j = 0; j < numGenes; j++) {
      encodings[batchOffset + j] = 0.0
    }

    // Inflate sparse or dense data
    if (isSparse) {
      // Handle sparse data format
      const start = indptr![cellIndex]
      const end = indptr![cellIndex + 1]
      for (let j = start; j < end; j++) {
        const geneIndex = indices![j]
        const targetIndex = inflationIndices[geneIndex]
        if (targetIndex !== -1) {
          encodings[batchOffset + targetIndex] = data[j]
        }
      }
    } else {
      // Handle dense data format
      for (let geneIndex = 0; geneIndex < sampleGenes.length; geneIndex++) {
        const targetIndex = inflationIndices[geneIndex]
        if (targetIndex !== -1) {
          encodings[batchOffset + targetIndex] = data[cellIndex * sampleGenes.length + geneIndex]
        }
      }
    }
  }
}

/**
 * Run the prediction on the model and store the results in IndexedDB
 */
async function predict(
  modelID: string,
  modelURL: string,
  h5File: File,
  cellRangePercent: number,
): Promise<void> {
  try {
    // Load the model via ONNX Runtime
    if (!self.model || self.model.modelID !== modelID) {
      self.model = await instantiateModel(modelURL, modelID)
    }

    // Reset our attention matrix to record attention for the top K genes
    // per class that explain the prediction
    const numClasses = self.model.classes.length
    const numGenes = self.model.genes.length
    self.attentionAccumulators = Array(numClasses)
      .fill(0)
      .map(() => Array(numGenes).fill(0))

    // Initialize the h5wasm file system
    const Module = await h5wasm.ready
    const { FS } = Module
    console.log('h5wasm loaded')
    FS.mount(FS.filesystems.WORKERFS, { files: [h5File] }, '/work')

    // Open the h5ad file
    self.postMessage({ type: 'status', message: 'Opening h5ad file...' })
    const h5 = new h5wasm.File(h5File.name, 'r')

    // Get various dataset attributes
    const obsNames = h5.get('obs/_index').value as string[]
    const varNames = h5.get('var/_index').value as string[]
    const shape = h5.get('X/shape').value

    // Determine if the data is stored in a sparse or dense format
    const isSparse = h5.get('X/data') !== null

    const totalNumCells = shape[0]
    const totalNumGenes = shape[1]
    console.log(`H5ad file has shape: ${totalNumCells} cells x ${totalNumGenes} genes`)

    // Apply cell range percent to limit the number of cells to process
    const numCellsToProcess = Math.floor((totalNumCells * cellRangePercent) / 100)
    console.log(`Processing ${numCellsToProcess} cells (${cellRangePercent}% of total)`)

    // Get the dataset - based on whether it's sparse or dense
    // If it's sparse use a CSR Matrix.
    let data, indices, indptr
    if (isSparse) {
      data = h5.get('X/data').value
      indices = h5.get('X/indices').value
      indptr = h5.get('X/indptr').value
    } else {
      data = h5.get('X').value
    }

    // Create inflation indices to map h5ad genes to model genes
    // Necessary since not all h5ad files have the same gene order
    const inflationIndices = precomputeInflationIndices(self.model.genes, varNames)

    // Create output arrays for predictions and probabilities
    const predictions: number[][] = []
    const probabilities: Float32Array[] = []

    // Create array to hold embeddings for UMAP
    const encodingsForUMAP: Float32Array[] = []
    const rng = new Prando(42)

    // Create a double buffer for batch encodings
    const encodingsBuffer = new Float32Array(self.batchSize * numGenes)

    // Process cells in batches
    const startTime = performance.now()
    let elapsedTime = 0

    for (let batchStart = 0; batchStart < numCellsToProcess; batchStart += self.batchSize) {
      const remainingCells = numCellsToProcess - batchStart
      const currentBatchSize = Math.min(remainingCells, self.batchSize)

      // Fill the batch data and inflate to model gene space
      fillBatchData(
        batchStart,
        currentBatchSize,
        data,
        indices,
        indptr,
        isSparse,
        varNames,
        inflationIndices,
        encodingsBuffer,
        numGenes,
      )

      // Create the tensor for the batch
      const tensor = new ort.Tensor(
        'float32',
        encodingsBuffer.slice(0, currentBatchSize * numGenes),
        [currentBatchSize, numGenes],
      )

      // Create the feeds for the model
      const feeds: Record<string, ort.Tensor> = { input: tensor }

      // Run the model
      const outputMap = await self.model.session.run(feeds)

      // Get the output tensors
      const logitsOutput = outputMap.logits.data as Float32Array
      const attendOutput = outputMap.attend.data as Float32Array
      const embeddingOutput = outputMap.embedding?.data as Float32Array

      // Accumulate attention for explaining predictions
      // The attention matrix is (batchSize, numClasses, numGenes)
      for (let i = 0; i < currentBatchSize; i++) {
        for (let classIndex = 0; classIndex < numClasses; classIndex++) {
          for (let geneIndex = 0; geneIndex < numGenes; geneIndex++) {
            const attendIndex = i * numClasses * numGenes + classIndex * numGenes + geneIndex
            self.attentionAccumulators![classIndex][geneIndex] += attendOutput[attendIndex]
          }
        }
      }

      // Process the output logits into predictions and probabilities
      for (let i = 0; i < currentBatchSize; i++) {
        const cellLogits = logitsOutput.slice(i * numClasses, (i + 1) * numClasses)
        const cellProbs = new Float32Array(cellLogits.length)

        // Convert logits to probabilities with softmax
        const maxLogit = Math.max(...cellLogits)
        let sumExp = 0
        for (let j = 0; j < cellLogits.length; j++) {
          cellProbs[j] = Math.exp(cellLogits[j] - maxLogit)
          sumExp += cellProbs[j]
        }
        for (let j = 0; j < cellProbs.length; j++) {
          cellProbs[j] /= sumExp
        }

        // Sort indices by probability (descending)
        const indices = Array.from({ length: numClasses }, (_, j) => j)
        indices.sort((a, b) => cellProbs[b] - cellProbs[a])

        // Store top 3 predictions and their probabilities
        predictions.push(indices.slice(0, 3))
        probabilities.push(cellProbs)

        // For UMAP, randomly sample cells
        if (
          embeddingOutput &&
          encodingsForUMAP.length < self.maxNumCellsToUMAP &&
          rng.next() < self.maxNumCellsToUMAP / numCellsToProcess
        ) {
          const embedding = embeddingOutput.slice(i * 128, (i + 1) * 128)
          encodingsForUMAP.push(new Float32Array(embedding))
        }
      }

      // Update progress
      elapsedTime = (performance.now() - startTime) / 60000
      self.postMessage({
        type: 'predictionProgress',
        message: `Processing cells (${elapsedTime.toFixed(2)} minutes elapsed)...`,
        countFinished: batchStart + currentBatchSize,
        totalToProcess: numCellsToProcess,
      })
    }

    // Calculate UMAP coordinates
    self.postMessage({ type: 'status', message: 'Calculating UMAP coordinates...' })

    let coords = new Float32Array(0)
    if (encodingsForUMAP.length > 0) {
      const umap = new UMAP.UMAP({
        nComponents: 2,
        nEpochs: 400,
        nNeighbors: 15,
      })

      // Convert array of Float32Arrays to a 2D array for UMAP
      const encodingsArray = encodingsForUMAP.map((arr) => Array.from(arr))

      // Fit and transform the data
      const embedding = umap.fit(encodingsArray)

      // Convert to flat array of [x1, y1, x2, y2, ...]
      coords = new Float32Array(embedding.length * 2)
      for (let i = 0; i < embedding.length; i++) {
        coords[i * 2] = embedding[i][0]
        coords[i * 2 + 1] = embedding[i][1]
      }
    }

    // Calculate the top genes per class based on accumulated attention
    const topGeneIndicesByClass: number[][] = []
    for (let classIndex = 0; classIndex < numClasses; classIndex++) {
      const classAttention = self.attentionAccumulators![classIndex]
      const indices = Array.from({ length: numGenes }, (_, i) => i)
      indices.sort((a, b) => classAttention[b] - classAttention[a])
      topGeneIndicesByClass.push(indices.slice(0, self.numExplainGenes))
    }

    // Calculate overall top genes across all classes
    const overallAttention = Array(numGenes).fill(0)
    for (let classIndex = 0; classIndex < numClasses; classIndex++) {
      for (let geneIndex = 0; geneIndex < numGenes; geneIndex++) {
        overallAttention[geneIndex] += self.attentionAccumulators![classIndex][geneIndex]
      }
    }
    const allIndices = Array.from({ length: numGenes }, (_, i) => i)
    allIndices.sort((a, b) => overallAttention[b] - overallAttention[a])
    const overallTopGenes = allIndices.slice(0, self.numExplainGenes)

    // Assign cell types based on top prediction
    const cellTypes = predictions.map((p) => p[0])

    // Store the results in IndexedDB
    self.postMessage({ type: 'status', message: 'Storing results in database...' })

    const db = await openDB('sims-web', 2)
    const tx = db.transaction('datasets', 'readwrite')
    const store = tx.objectStore('datasets')

    const result: PredictionResult = {
      datasetLabel: h5File.name,
      coords,
      cellTypeNames: self.model.classes,
      cellTypes,
      modelID: self.model.modelID,
      topGeneIndicesByClass,
      genes: self.model.genes,
      overallTopGenes,
      cellNames: obsNames.slice(0, numCellsToProcess),
      predictions,
      probabilities,
    }

    await store.add(result)
    await tx.done
    db.close()

    // Clean up
    h5.close()
    h5wasm.fs.unmount(root)

    // Notify main thread of completion
    self.postMessage({
      type: 'finishedPrediction',
      totalProcessed: numCellsToProcess,
      totalNumCells,
      elapsedTime,
    })
  } catch (error) {
    console.error('Prediction error:', error)
    self.postMessage({
      type: 'predictionError',
      error: error instanceof Error ? error.message : String(error),
    })
  }
}

export default {} as typeof Worker & { new (): Worker }
