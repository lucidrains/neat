import neat
import numpy as np

inputs  = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
targets = np.array([[0],    [1],    [1],    [0]],    dtype=np.float32)

net = neat.NEAT(2, 4, 1, pop_size=1)

for epoch in range(2000):
    loss = 0.0
    for i in range(4):
        pred = net.single_forward(0, inputs[i])
        loss += 0.5 * (pred[0] - targets[i][0]) ** 2
        net.backprop(0, inputs[i], targets[i], learning_rate=0.1)

print(f"loss: {loss:.6f}")
for i in range(4):
    print(f"  {inputs[i].tolist()} -> {net.single_forward(0, inputs[i])[0]:.6f}")
