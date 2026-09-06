# nimporter_plus transparently compiles Nim extensions on import.

test-xor:
	python3 train_neat_xor.py

train-lunar-fuss:
	uv run train_lunar.py --use_fuss=True --fuss_eps=1e-5

# sync the version across pyproject.toml and neat.nimble

bump version:
	python3 bump.py {{version}}
