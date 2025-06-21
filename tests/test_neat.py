from neat.neat import add_neuron, add_synapse

def test_add_neuron_and_synapse():
    assert add_neuron() == 0
    assert add_neuron() == 1
    assert add_synapse(0, 1) == 0
