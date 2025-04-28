predict:
	python scripts/cluster.py predict \
		public/models/allen-celltypes+human-cortex+various-cortical-areas.onnx \
		checkpoints/allen-celltypes+human-cortex+various-cortical-areas.h5ad
		
train:
	python scripts/cluster.py train \
		public/models/allen-celltypes+human-cortex+various-cortical-areas-encodings.npy

map:
	python scripts/cluster.py map \
		public/models/allen-celltypes+human-cortex+various-cortical-areas-pumap.onnx \
		public/models/allen-celltypes+human-cortex+various-cortical-areas-encodings.npy