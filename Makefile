predict:
	# python scripts/cluster.py predict \
	# 	public/models/allen-celltypes+human-cortex+various-cortical-areas.onnx \
	# 	checkpoints/allen-celltypes+human-cortex+various-cortical-areas.h5ad
	python scripts/cluster.py predict \
		public/models/pre-postnatal-cortex+all+rna.onnx \
		checkpoints/pre-postnatal-cortex+all+rna.h5ad \
		--cell-type-field Cell_Type
		
train:
	# python scripts/cluster.py train \
	# 	public/models/allen-celltypes+human-cortex+various-cortical-areas-encodings.npy
	python scripts/cluster.py train \
		public/models/pre-postnatal-cortex+all+rna-encodings.npy

map:
	# python scripts/cluster.py map \
	# 	public/models/allen-celltypes+human-cortex+various-cortical-areas-pumap.onnx \
	# 	public/models/allen-celltypes+human-cortex+various-cortical-areas-encodings.npy
	# python scripts/cluster.py map \
	# 	public/models/pre-postnatal-cortex+all+rna-pumap.onnx \
	# 	public/models/pre-postnatal-cortex+all+rna-encodings.npy