/**
 * Browser Web Worker that runs UMAP on encodings stored in IndexedDB.
 *
 * To start UMAP computation, the worker expects a message with the following data:
 * {
 *   type: "startUMAP",
 *   datasetLabel: "dataset_label",
 * }
 *
 * The worker will then retrieve the encodings from IndexedDB, run UMAP,
 * and update the coords field in IndexedDB. It will send "processingProgress" messages to the
 * main thread to update the progress bar and "finishedUMAP" message when done.
 * If an error occurs, it will send an "umapError" message with the error.
 */
import Prando from 'prando'
import * as UMAP from 'umap-js'
import { openDB } from 'idb'

// Define TypeScript interfaces for the worker's data structures
interface UMAPMessage {
  type: 'startUMAP'
  datasetLabel: string
}

interface StatusMessage {
  type: 'status'
  message: string
}

interface ProgressMessage {
  type: 'processingProgress'
  message: string
  countFinished: number
  totalToProcess: number
}

interface FinishedMessage {
  type: 'finishedUMAP'
  datasetLabel: string
  elapsedTime: number
}

interface ErrorMessage {
  type: 'umapError'
  error: any
}

type WorkerMessage = UMAPMessage
type MainThreadMessage = StatusMessage | ProgressMessage | FinishedMessage | ErrorMessage

// Declare worker scope variables with proper types
declare const self: DedicatedWorkerGlobalScope

// Handle messages from the main thread
self.addEventListener('message', async function (event: MessageEvent<WorkerMessage>) {
  if (event.data.type === 'startUMAP') {
    computeUMAP(event.data.datasetLabel)
  }
})

/**
 * Compute UMAP coordinates from encodings stored in IndexedDB
 * @param {string} datasetLabel - The label of the dataset to process
 */
async function computeUMAP(datasetLabel: string): Promise<void> {
  const startTime = Date.now() // Record start time

  try {
    self.postMessage({ type: 'status', message: 'Starting UMAP computation...' })

    // Retrieve the dataset from IndexedDB
    const db = await openDB('sims-web')
    const tx = db.transaction('datasets', 'readonly')
    const store = tx.objectStore('datasets')
    const dataset = await store.get(datasetLabel)
    await tx.done

    if (!dataset || !dataset.encodings) {
      throw new Error('Dataset or encodings not found in IndexedDB')
    }

    // Initialize UMAP with a fixed random seed for reproducibility
    const prando = new Prando(42)
    const random = () => prando.next()

    const umap = new UMAP.UMAP({
      random,
      nComponents: 2,
      nEpochs: 400,
      nNeighbors: 15,
    })

    // Run UMAP with progress updates
    let coordinates: number[][] | null = null
    try {
      coordinates = await umap.fitAsync(dataset.encodings, (epochNumber: number) => {
        // Update progress and give user feedback
        self.postMessage({
          type: 'processingProgress',
          message: `Computing UMAP coordinates...`,
          countFinished: epochNumber,
          totalToProcess: umap.getNEpochs(),
        })
      })
    } catch (error) {
      self.postMessage({ type: 'umapError', error })
      throw error
    }

    // Update the dataset in IndexedDB with the new coordinates
    const updateTx = db.transaction('datasets', 'readwrite')
    const updateStore = updateTx.objectStore('datasets')

    // Get the latest version of the dataset
    const updatedDataset = await updateStore.get(datasetLabel)

    // Update the coordinates
    updatedDataset.coords = coordinates!.flat()

    // Put the updated dataset back in the store
    await updateStore.put(updatedDataset)
    await updateTx.done
    db.close()

    // Calculate elapsed time
    const endTime = Date.now()
    const elapsedTime = (endTime - startTime) / 60000 // Calculate elapsed time in minutes

    // Let the main thread know we're done
    self.postMessage({
      type: 'finishedUMAP',
      datasetLabel: datasetLabel,
      elapsedTime,
    })
  } catch (error) {
    self.postMessage({ type: 'umapError', error: error })
  }
}
