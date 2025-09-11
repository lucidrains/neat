# Claude.md - Project Context for NEAT

## Project Overview
NEAT (NeuroEvolution of Augmenting Topologies) implementation with HyperNEAT support, using JAX for performance and Nim for critical algorithms.

## Key Components
- **neat/neat.py**: Main NEAT implementation in Python/JAX
- **neat/neat_nim.nim**: High-performance Nim extensions for speciation and other algorithms
- **tests/**: Test suite for NEAT and HyperNEAT functionality
- **train_lunar.py**: Example training script for Lunar Lander environment

## Development Commands

### Setup and Installation
```bash
# Install in development mode
pip install -e .

# Install with gym dependencies for reinforcement learning
pip install -e ".[gym]"
```

### Testing
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test
python -m pytest tests/test_neat.py::test_neat -v

# Run Nim tests
nimble test
```

### Linting and Type Checking
```bash
# Python linting (if configured)
# Add linting commands here when configured

# Nim checks
nim check neat/neat_nim.nim
```

### Training Examples
```bash
# Train on Lunar Lander
python train_lunar.py
```

## Project Structure
- Uses JAX for GPU-accelerated neural network operations
- Nim extensions for performance-critical algorithms (speciation)
- Supports both NEAT and HyperNEAT variants
- Includes gymnasium integration for RL environments

## Dependencies
- Python >= 3.9
- Nim compiler (for building extensions)
- JAX/JAXlib for neural network operations
- nimporter for Python-Nim integration
- gymnasium for RL environments (optional)

## Notes
- The project uses nimporter to seamlessly integrate Nim code with Python
- JAX is used for vectorized operations and potential GPU acceleration
- Test coverage includes both NEAT and HyperNEAT implementations

## Style
Prioritize simple, concise, pure functions. Think ultrahard about whether the solution is too complicated before getting back to me. Please follow my exact instructions and give me the minimal code, no fluff or anything extraneous

always use snake case for variable and function names in Nim (Nim is agnostic to snake case or camel case)