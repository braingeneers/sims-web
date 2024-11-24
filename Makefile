check-model:
	python -m onnxruntime.tools.check_onnx_model_mobile_usability --log_level debug data/sims.onnx

deploy:
	rsync -v ./*.{js,html} rcurrie@park.gi.ucsc.edu:~/public_html/sims/
	rsync -avz models rcurrie@park.gi.ucsc.edu:~/public_html/sims/
	rsync -avz data rcurrie@park.gi.ucsc.edu:~/public_html/sims/