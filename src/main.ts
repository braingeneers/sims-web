import { createApp } from 'vue'
import { createVuetify } from 'vuetify'
import * as components from 'vuetify/components'
import * as directives from 'vuetify/directives'
import '@fontsource/roboto'

import App from './App.vue'

const vuetify = createVuetify({
  components,
  directives,
  theme: {
    defaultTheme: 'dark'
  },
  defaults: {
    VApp: {
      style: 'font-family: Roboto, sans-serif;'
    }
  }
})

createApp(App).use(vuetify).mount('#app')
