## neat (wip)

Explorations into NEAT and some of its derivative research

## install

In project root, run

```bash
$ sh install.sh
```

## usage

Distill / behavior-clone an evolved champion NEAT network into any barebones PyTorch MLP:

```python
import torch
import torch.nn as nn
import gymnasium as gym

from neat import NEAT, behavior_clone

# 1. train NEAT on an environment

env = gym.make('CartPole-v1')
pop = NEAT(4, 16, 2, pop_size = 64)

# ... evolve population ...

# 2. extract champion network

champion = pop.champion

# 3. define any barebones PyTorch MLP

mlp = nn.Sequential(
    nn.Linear(4, 32),
    nn.Tanh(),
    nn.Linear(32, 2)
)

# 4. behavior clone and save .pt to project root

behavior_clone(
    env,
    champion,
    mlp,
    save_path = 'mlp-cartpole.pt'  # saved directly to project root
)
```

## quick test

```bash
$ uv run train_lunar.py
```

To run the end-to-end CartPole evolution, behavior cloning, and validation:

```bash
$ uv run train_bc_cartpole.py
```

## citations

```bibtex
@article{Stanley2011CompetitiveCT,
    title   = {Competitive Coevolution through Evolutionary Complexification},
    author  = {Kenneth O. Stanley and Risto Miikkulainen},
    journal = {ArXiv},
    year    = {2011},
    volume  = {abs/1107.0037},
    url     = {https://api.semanticscholar.org/CorpusID:11881625}
}
```

```bibtex
@inproceedings{4665912,
    author  = {Miguel, Cesar Gomes and Silva, Carolina Feher da and Netto, Marcio Lobo},
    booktitle = {2008 10th Brazilian Symposium on Neural Networks},
    title   = {Structural and Parametric Evolution of Continuous-Time Recurrent Neural Networks},
    year    = {2008},
    doi     = {10.1109/SBRN.2008.12}
}
```

```bibtex
@article{Khamesian2021HybridSN,
    title   = {Hybrid self-attention NEAT: a novel evolutionary self-attention approach to improve the NEAT algorithm in high dimensional inputs},
    author  = {Saman Khamesian and Hamed Malek},
    journal = {Evolving Systems},
    year    = {2021},
    pages   = {1-15},
    url     = {https://api.semanticscholar.org/CorpusID:244920723}
}
```

```bibtex
@article{Hornby2006AutomatedAD,
    title   = {Automated Antenna Design with Evolutionary Algorithms},
    author  = {Gregory Hornby and Al Globus and Derek S. Linden and Jason D. Lohn},
    journal = {Space},
    year    = {2006},
    url     = {https://api.semanticscholar.org/CorpusID:8290212}
}
```

```bibtex
@inproceedings{schrum:gecco14,
    title   = {Evolving Multimodal Behavior With Modular Neural Networks in Ms. Pac-Man},
    author  = {Jacob Schrum and Risto Miikkulainen},
    booktitle = {Proceedings of the Genetic and Evolutionary Computation Conference (GECCO 2014)},
    month   = {July},
    address = {Vancouver, BC, Canada},
    pages   = {325--332},
    note    = {Best Paper: Digital Entertainment and Arts},
    url     = {http://www.cs.utexas.edu/users/ai-lab?schrum:gecco2014},
    year    = {2014}
```

```bibtex
@article{stanley:ec02,
    title   = {Evolving Neural Networks Through Augmenting Topologies},
    author  = {Kenneth O. Stanley and Risto Miikkulainen},
    volume  = {10},
    journal = {Evolutionary Computation},
    number  = {2},
    pages   = {99-127},
    url     = "http://nn.cs.utexas.edu/?stanley:ec02",
    year    = {2002}
}
```

```bibtex
@misc{doerr2017fastgeneticalgorithms,
    title   = {Fast Genetic Algorithms},
    author  = {Benjamin Doerr and Huu Phuoc Le and Régis Makhmara and Ta Duy Nguyen},
    year    = {2017},
    eprint  = {1703.03334},
    archivePrefix = {arXiv},
    primaryClass = {cs.NE},
    url     = {https://arxiv.org/abs/1703.03334},
}
```

```bibtex
@misc{legg2004tournamentversusfitnessuniform,
    title   = {Tournament versus Fitness Uniform Selection},
    author  = {Shane Legg and Marcus Hutter and Akshat Kumar},
    year    = {2004},
    eprint  = {cs/0403038},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url     = {https://arxiv.org/abs/cs/0403038},
}
```

```bibtex
@article{hiraga2024improving,
    title   = {Improving the performance of mutation-based evolving artificial neural networks with self-adaptive mutations},
    author  = {Hiraga, Motoaki and Komura, Masahiro and Miyamoto, Akiharu and Morimoto, Daichi and Ohkura, Kazuhiro},
    journal = {PLOS ONE},
    volume  = {19},
    number  = {7},
    pages   = {e0307084},
    year    = {2024},
    publisher = {Public Library of Science},
    doi     = {10.1371/journal.pone.0307084},
    url     = {https://doi.org/10.1371/journal.pone.0307084}
}
```

```bibtex
@inproceedings{tackett1994unique,
    title     = {The unique implications of brood selection for genetic programming},
    author    = {Tackett, Walter Alden and Carmi, Aviram},
    booktitle = {Proceedings of the First IEEE Conference on Evolutionary Computation. IEEE World Congress on Computational Intelligence},
    pages     = {160--165},
    year      = {1994},
    organization = {IEEE},
    doi       = {10.1109/ICEC.1994.350030}
}
```

```bibtex
@article{clune2013evolutionary,
    title     = {The evolutionary origins of modularity},
    author    = {Clune, Jeff and Mouret, Jean-Baptiste and Lipson, Hod},
    journal   = {Proceedings of the Royal Society B: Biological Sciences},
    volume    = {280},
    number    = {1755},
    pages     = {20122863},
    year      = {2013},
    publisher = {The Royal Society},
    doi       = {10.1098/rspb.2012.2863}
}
```
