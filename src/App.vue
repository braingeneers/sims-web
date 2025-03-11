<template>
  <v-app>
    <v-app-bar>
      <v-toolbar-title>Cell Stray</v-toolbar-title>
      <v-spacer></v-spacer>
      <v-btn @click="toggleTheme">
        <v-icon>mdi-theme-light-dark</v-icon>
      </v-btn>
    </v-app-bar>
    <v-main>
      <div>Cell Stray</div>
    </v-main>
  </v-app>
</template>

<script setup lang="ts">
import { useTheme } from 'vuetify'
const theme = useTheme()

function toggleTheme() {
  theme.global.name.value = theme.global.current.value.dark ? 'light' : 'dark'
}
</script>

<style>
@import '@mdi/font/css/materialdesignicons.min.css';
</style>
