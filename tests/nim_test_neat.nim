import std/[
  random,
  assertions
]

import ../neat/neat_nim

proc test() =
  
  randomize(42)
  
  let fitnesses = @[1.0'f32, 5.0'f32, 3.0'f32, 8.0'f32, 2.0'f32]
  let result = tournament(fitnesses, num_tournaments=1, tournament_size=3)
  
  assert result.len == 1
  let couple = result[0]
  
  assert couple.parent1.fitness >= couple.parent2.fitness
  assert couple.parent1.index >= 0 and couple.parent1.index < fitnesses.len
  assert couple.parent2.index >= 0 and couple.parent2.index < fitnesses.len
  assert couple.parent1.index != couple.parent2.index

when is_main_module:
  test()
