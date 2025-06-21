from neat.neat import (
    add_neat_topology,
    add_neuron,
    add_synapse
)

def test_add_neuron_and_synapse():
    top_id = add_neat_topology()
    assert add_neuron(top_id) == 0
    assert add_neuron(top_id) == 1
    assert add_synapse(top_id, 0, 1) == 0
