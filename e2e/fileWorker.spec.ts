import { test, expect } from '@playwright/test'

// Configure test to use proper origin
// const workerUrl = 'http://localhost:5173/src/workers/fileWorker.js'
const workerUrl = '/src/workers/fileWorker.js'

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
