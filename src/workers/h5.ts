self.onmessage = function (e) {
  const { h5File } = e.data
  console.log('h5File', h5File)
  self.postMessage({
    type: 'buffer',
    buffer: 5,
  })
}
