import webbrowser
from aocd.models import Puzzle

from intcode import *

def part1(input):
    program = parse_program(input)
    return IntcodeVM(program).run(1)

def part2(input):
    program = parse_program(input)
    return IntcodeVM(program).run(2)

def main():
    puzzle = Puzzle(year=2019, day=9)
    print("Part 1", part1(puzzle.input_data))
    print("Part 2", part2(puzzle.input_data))

    # Part 1 3409270027
    # Part 2 82760

if __name__ == "__main__":
    main()
