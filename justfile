rebuild:
	rm -f neat/neat_nim.*.so
	rm -rf build/
	rm -rf nim-extensions/
	rm -rf neat/__pycache__/
	uv pip install -e .
