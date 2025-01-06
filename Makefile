serve:
	python -m http.server 3000

check-model:
	python -m onnxruntime.tools.check_onnx_model_mobile_usability --log_level debug public/models/default.onnx

deploy:
	git checkout gh-pages
	git merge main
	git push origin gh-pages
	git checkout main