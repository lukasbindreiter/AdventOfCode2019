from aocd.models import Puzzle
from itertools import permutations, cycle

from intcode import *

def part1(input):
    program = parse_program(input)

    robot_pos = 0 + 0j  # y + x
    robot_dir = -1 + 0j  # facing upwards
    dirs = {0: 1j, 1: -1j}

    tiles = {}
    vm = IntcodeVM(program)
    while not vm.stopped:
        color = tiles.get(robot_pos, 0)
        color, turn = vm.run(color)
        tiles[robot_pos] = color
        robot_dir *= dirs[turn]
        robot_pos += robot_dir
    return len(tiles)


def part2(input):
    program = parse_program(input)

    def y(pos):
        return int(np.real(pos))

    def x(pos):
        return int(np.imag(pos))

    robot_pos = 0 + 0j  # y + x
    robot_dir = -1 + 0j  # facing upwards
    dirs = {0: 1j, 1: -1j}

    tiles = {robot_pos: 1}
    vm = IntcodeVM(program)
    while not vm.stopped:
        color = tiles.get(robot_pos, 0)
        color, turn = vm.run(color)
        tiles[robot_pos] = color
        robot_dir *= dirs[turn]
        robot_pos += robot_dir

    # start at 0, 0
    min_y = min(y(pos) for pos in tiles)
    min_x = min(x(pos) for pos in tiles)
    tiles = {(y(pos) - min_y, x(pos) - min_x): tiles[pos] for pos in tiles}
    max_y = max(y for y, x in tiles)
    max_x = max(x for y, x in tiles)

    result = np.zeros(shape=(max_y + 1, max_x + 1), dtype=np.int)
    for y, x in tiles:
        result[y, x] = tiles[(y, x)]

    def pprint(color):
        return " " if color == 0 else "█"

    return "\n".join("".join(map(pprint, line)) for line in result)


def main():
    puzzle = Puzzle(year=2019, day=11)
    print("Part 1", part1(puzzle.input_data))
    print("Part 2\n" + part2(puzzle.input_data))

    # Part 1 2336
    # Part 2 UZAEKBLP
    # █  █ ████  ██  ████ █  █ ███  █    ███
    # █  █    █ █  █ █    █ █  █  █ █    █  █
    # █  █   █  █  █ ███  ██   ███  █    █  █
    # █  █  █   ████ █    █ █  █  █ █    ███
    # █  █ █    █  █ █    █ █  █  █ █    █
    #  ██  ████ █  █ ████ █  █ ███  ████ █


if __name__ == "__main__":
    main()
