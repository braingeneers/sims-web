# Benchmarks

These are more notes at the moment then actual benchmarks...

## Latest version with dynamic batch size, multi-threaded ONNX and double buffering of I/O

#threads/batch size
10/9 Processed 1759 of 175925 cells in 0.26 minutes
10/18 Processed 1759 of 175925 cells in 0.24 minutes
10/45 Processed 1759 of 175925 cells in 0.21 minutes
10/180 Processed 1759 of 175925 cells in 0.19 minutes
10/450 Processed 1759 of 175925 cells in 0.18 minutes

Larger I/O via h5wasm improves time, but higher memory usage

## Inference only on the same expression vector on first 1% of rnh027_log1p_only.h5ad on Nov 2023 MacBook M3 Pro w/18GB

Batch Size = 1

proxy = false
1 Thread
Processed 1759 of 175925 cells in 0.67 minutes

4 Threads (equivalent to maxThreads = 0 which let's ONNX pick #)
Processed 1759 of 175925 cells in 0.58 minutes

8 Threads
Processed 1759 of 175925 cells in 0.61 minutes

proxy = true
1 Thread
Processed 1759 of 175925 cells in 0.67 minutes

8 Threads
Processed 1759 of 175925 cells in 0.62 minutes

16 Threads
Processed 1759 of 175925 cells in 1.09 minutes

## Batch worker.js

First pass that runs and partially matches results
Single Threaded Processed 1759 of 175925 cells in 0.88 minutes
vs.
11 Threads Processed 1759 of 175925 cells in 0.31 minutes

## Pure inference with no h5 i/o

1024 samples in 0.3834 minutes with a batch size of 1 and 1 threads
1024 samples in 0.0780 minutes with a batch size of 8 and 8 threads

2048 samples in 0.2722 minutes with a batch size of 16 and 16 threads
2048 samples in 0.1482 minutes with a batch size of 32 and 8 threads
2048 samples in 0.1625 minutes with a batch size of 11 and 11 threads
2048 samples in 0.1523 minutes with a batch size of 11 and 11 threads
2048 samples in 0.1583 minutes with a batch size of 8 and 8 threads
2048 samples in 0.2228 minutes with a batch size of 4 and 4 threads
2048 samples in 0.7652 minutes with a batch size of 1 and 1 threads
