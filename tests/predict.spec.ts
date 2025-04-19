import { test, expect } from '@playwright/test'

test('Navigate to root, click run, wait for results', async ({ page }) => {
  // Change the URL if your app runs on a different port or host.
  await page.goto('/')

  // Click the run button.
  // In your App.vue the run button is a v-app-bar-nav-icon with icon="mdi-play".
  // You might want to add a data attribute (e.g., data-cy="run-button") to ease selection.
  // For now, we'll target it by its color property if possible.
  await page.click('[data-cy="run-button"]')

  // Wait for the status element to display the expected value.
  // Replace '[data-cy=status-label]' and 'Expected Value' with your actual selector and value.
  await expect(page.locator('[data-cy=results]')).toHaveText('Cells: 100 Genes: 33694')
})
