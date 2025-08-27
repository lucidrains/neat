import unittest2
import std/random

import ../neat/neat_nim

suite "Neat":
  
  setup:
    randomize(42)
  
  test "tournament selects two highest fitness individuals from tournament":
    let fitnesses = @[1.0'f32, 5.0'f32, 3.0'f32, 8.0'f32, 2.0'f32]
    let result = tournament(fitnesses, num_tournaments=1, tournament_size=3)
    
    check result.len == 1
    let couple = result[0]
    
    check couple.parent1.fitness >= couple.parent2.fitness
    check couple.parent1.index >= 0 and couple.parent1.index < fitnesses.len
    check couple.parent2.index >= 0 and couple.parent2.index < fitnesses.len
    check couple.parent1.index != couple.parent2.index
