check-model:
	python -m onnxruntime.tools.check_onnx_model_mobile_usability --log_level debug data/sims.onnx
