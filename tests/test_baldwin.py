import numpy as np
import torch

from neat.baldwin import BaldwinNEAT, PredictiveCodingMemory


def test_dead_simple_usage():
    pop = BaldwinNEAT(4, 2, pop_size = 16, num_memory = 4, memory_lr = 0.1, memory_seed = 0)

    obs = np.zeros((16, 4), dtype = np.float32)
    actions = pop.forward(obs)
    loss = pop.observe_transition(obs, obs + 0.1)

    assert actions.shape == (16, 2)
    assert loss > 0.0
    assert pop.memory.memory.shape == (16, 4)
    assert torch.any(pop.memory.memory != 0.0)

    pop.reset_episode()
    assert torch.all(pop.memory.memory == 0.0)

    pop.set_ttt(False)
    assert pop.observe_transition(obs, obs + 0.1) is None

    pop.set_ttt(True)
    assert pop.observe_transition(obs, obs + 0.1) is not None


def test_usage_loop():
    pop = BaldwinNEAT(4, 2, pop_size = 16, num_memory = 4, memory_seed = 0)
    pop.reset_recurrent_state()

    for _ in range(8):
        pop.reset_episode()
        obs = np.zeros((16, 4), dtype = np.float32)

        for _ in range(4):
            actions = pop.forward(obs)
            next_obs = obs + 0.1
            pop.observe_transition(obs, next_obs)
            obs = next_obs

    assert actions.shape == (16, 2)
    assert torch.isfinite(pop.memory.memory).all()


def test_memory_updates_only_designated_nodes():
    pc = PredictiveCodingMemory(3, pop_size = 2, num_memory = 2, lr = 0.1, seed = 0)

    obs = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype = np.float32)
    next_obs = obs + np.array([[0.5, -0.25, 0.1], [0.2, 0.0, -0.3]], dtype = np.float32)

    features = pc.observe(obs)
    assert features.shape == (2, 5)
    assert np.allclose(features[:, 3:], 0.0)

    loss = pc.update(obs, next_obs)
    assert loss is not None and np.isfinite(loss)
    assert np.any(pc.memory.numpy() != 0.0)

    # predictor weights are buffers: gradients only reach the memory leaves
    assert pc.weights.requires_grad is False
    assert pc.weights.grad is None

    features = pc.observe(obs)
    assert np.allclose(features[:, 3:], pc.memory.numpy())


def test_memory_can_be_frozen():
    pc = PredictiveCodingMemory(2, pop_size = 2, num_memory = 2, lr = 0.1, ttt = False, seed = 0)
    obs = np.ones((2, 2), dtype = np.float32)

    assert pc.update(obs, obs * 2.0) is None
    assert np.allclose(pc.memory.numpy(), 0.0)

    pc.set_ttt(True)
    assert pc.update(obs, obs * 2.0) is not None
    assert np.any(pc.memory.numpy() != 0.0)


def test_baldwin_neat_forward_and_episode_reset():
    neat = BaldwinNEAT(3, 4, 2, pop_size = 4, num_memory = 2, memory_seed = 0)

    obs = np.zeros((4, 3), dtype = np.float32)
    assert neat.forward(obs).shape == (4, 2)

    assert neat.observe_transition(obs, obs + 1.0) is not None
    assert torch.any(neat.memory.memory != 0.0)

    neat.reset_episode()
    assert torch.all(neat.memory.memory == 0.0)


def test_carry_across_episodes_keeps_memory():
    neat = BaldwinNEAT(
        3, 4, 2,
        pop_size = 4,
        num_memory = 2,
        carry_across_episodes = True,
        memory_seed = 0
    )

    obs = np.zeros((4, 3), dtype = np.float32)
    neat.observe_transition(obs, obs + 1.0)
    before = neat.memory.memory.clone()

    neat.reset_episode()
    assert torch.allclose(neat.memory.memory, before)


def test_frozen_neat_keeps_memory_zero():
    neat = BaldwinNEAT(3, 4, 2, pop_size = 4, num_memory = 2, ttt = False, memory_seed = 0)

    obs = np.zeros((4, 3), dtype = np.float32)
    assert neat.observe_transition(obs, obs + 1.0) is None
    assert torch.all(neat.memory.memory == 0.0)
