from aocd.models import Puzzle
from itertools import permutations, cycle

from intcode import *


def run_sequential(phases, program):
    signal = 0
    for phase in phases:
        signal = IntcodeVM(program).run([phase, signal])
    return signal


def part1(input):
    program = parse_program(input)
    return max(run_sequential(phases, program) for phases in permutations(range(5)))


def run_continous(phases, program):
    vms = [IntcodeVM(program) for _ in phases]
    for phase, vm in zip(phases, vms):
        vm.run(phase)
    
    signal = 0
    finished = 0
    for vm in cycle(vms):
        if finished == len(vms):
            break
        signal = vm.run(signal)
        if vm.stopped:
            finished += 1

    return signal


def part2(input):
    program = parse_program(input)
    return max(run_continous(phases, program) for phases in permutations(range(5, 10)))


def main():
    puzzle = Puzzle(year=2019, day=7)
    print("Part 1", part1(puzzle.input_data))
    print("Part 2", part2(puzzle.input_data))

    # Part 1 30940
    # Part 2 76211147


if __name__ == "__main__":
    main()
