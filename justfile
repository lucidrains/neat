# nimporter caches aggressively. If python imports don't reflect .nim changes,
# run `just rebuild` to nuke caches and recompile.

rebuild:
	rm -f neat/neat_nim.*.so
	rm -rf build/
	rm -rf nim-extensions/
	rm -rf neat/__pycache__/
	uv pip install -e .

test-xor: rebuild
	python3 train_neat_xor.py
