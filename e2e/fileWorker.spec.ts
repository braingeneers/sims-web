import { test, expect } from '@playwright/test'

const workerUrl = '/src/workers/PredictWorker.js'
// const workerUrl = '/src/worker.js'

test('Web Worker doubles the input number', async ({ page }) => {
  await page.goto('/')

  const workerResult = await page.evaluate(async (url) => {
    return new Promise((resolve) => {
      const worker = new Worker(url)
      worker.onmessage = (event) => {
        resolve(event.data)
        worker.terminate()
      }
      worker.postMessage(5)
    })
  }, workerUrl)

  expect(workerResult).toBe(10)
})
