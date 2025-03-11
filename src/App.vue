<template>
  <v-app>
    <v-app-bar>
      <v-toolbar-title>Cell Dimension</v-toolbar-title>
      <v-spacer></v-spacer>
      <v-btn @click="openFileDialog = true">
        <v-icon>mdi-folder-open</v-icon>
      </v-btn>
      <v-btn @click="toggleTheme">
        <v-icon>mdi-theme-light-dark</v-icon>
      </v-btn>
    </v-app-bar>
    <v-main>
      <div>Cell Stray</div>
    </v-main>
    <v-dialog v-model="openFileDialog" width="500">
      <v-card>
        <v-card-title>Open File</v-card-title>
        <v-card-actions>
          <v-spacer></v-spacer>
          <v-btn @click="openFileDialog = false">Close</v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
  </v-app>
</template>

<script setup lang="ts">
import { useTheme } from 'vuetify'
import { ref } from 'vue'

const theme = useTheme()
const openFileDialog = ref(false)

function toggleTheme() {
  theme.global.name.value = theme.global.current.value.dark ? 'light' : 'dark'
}
</script>

<style>
@import '@mdi/font/css/materialdesignicons.min.css';

:root {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
}

.v-application {
  font-family: inherit;
  line-height: 1.6;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

.v-toolbar-title {
  font-weight: 600;
}
</style>
