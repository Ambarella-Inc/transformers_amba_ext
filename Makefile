
build:
	echo "Build_wheel_package:"
	python3 setup.py bdist_wheel

clean:
	echo "Clean_wheel_package:"
	find . -name '__pycache__' -type d -exec rm -rf {} +
	find . -name '*egg-info' -type d -exec rm -rf {} +
	rm -rf build dist

doc:
	sphinx-apidoc -o docs/source/ src/transformers_amba_ext \
		"src/transformers_amba_ext/generate"                \
		"src/transformers_amba_ext/inference"               \
		"src/transformers_amba_ext/utils"                   \
		"src/transformers_amba_ext/models/llava/llava"      \
		"src/transformers_amba_ext/models/llava_onevision/llava_next"
	cd docs; make clean; make html