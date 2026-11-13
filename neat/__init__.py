# single source of truth for the version - bump with `just bump <version>`
__version__ = "0.0.16"

from neat.neat import (
    NEAT,
    Topology,
    Network,
    behavior_clone,
    TaskRouter,
    get_population_connection_costs,
    get_population_modularity_q
)

from neat.baldwin import BaldwinNEAT, PredictiveCodingMemory

