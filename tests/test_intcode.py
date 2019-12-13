import numpy as np
import pytest
from itertools import cycle
from aocd.models import Puzzle

from aoc.intcode import *


def get_program(day: int) -> np.array:
    return parse_program(Puzzle(year=2019, day=day).input_data)


# fmt: off
@pytest.mark.parametrize(["program", "expected_output"], [
    ((1,9,10,3,2,3,11,0,99,30,40,50), (3500,9,10,70,2,3,11,0,99,30,40,50)),
    ((1,0,0,0,99), (2,0,0,0,99)),
    ((2,3,0,3,99), (2,3,0,6,99)),
    ((2,4,4,5,99,0), (2,4,4,5,99,9801)),
    ((1,1,1,4,99,5,6,0,99), (30,1,1,4,2,5,6,0,99))
])
# fmt: on
def test_day2_part1_examples(program, expected_output):
    program = np.array(program)
    vm = IntcodeVM(program)
    vm.run()
    assert np.allclose(vm.memory, expected_output)

def test_day2_part1():
    program = get_program(day=2)
    program[1] = 12
    program[2] = 2
    vm = IntcodeVM(program)
    vm.run()
    assert vm.memory[0] == 3760627

def test_day2_part2():
    program = get_program(day=2)
    program[1] = 71
    program[2] = 95
    vm = IntcodeVM(program)
    vm.run()
    assert vm.memory[0] == 19690720

def test_day5_part1():
    program = get_program(day=5)
    diagnostic = IntcodeVM(program).run(1)[-1]
    assert diagnostic == 5821753

def test_day5_part2_example():
    # fmt: off
    program = np.array([3,21,1008,21,8,20,1005,20,22,107,8,21,20,1006,20,31,1106,0,36,98,0,0,1002,21,125,20,4,20,1105,1,46,104,999,1105,1,46,1101,1000,1,20,4,20,1105,1,46,98,99])
    # fmt: on
    assert IntcodeVM(program).run(7) == 999
    assert IntcodeVM(program).run(8) == 1000
    assert IntcodeVM(program).run(9) == 1001
    
def test_day5_part2():
    program = get_program(day=5)
    diagnostic = IntcodeVM(program).run(5)
    assert diagnostic == 11956381

# fmt: off
@pytest.mark.parametrize(["program", "phases", "expected_output"], [
    ((3,15,3,16,1002,16,10,16,1,16,15,15,4,15,99,0,0), (4,3,2,1,0), 43210),
    ((3,23,3,24,1002,24,10,24,1002,23,-1,23,101,5,23,23,1,24,23,23,4,23,99,0,0), (0,1,2,3,4), 54321),
    ((3,31,3,32,1002,32,10,32,1001,31,-2,31,1007,31,0,33,1002,33,7,33,1,33,31,31,1,32,31,31,4,31,99,0,0,0), (1,0,4,3,2), 65210)
])
# fmt: on
def test_day7_part1_examples(program, phases, expected_output):
    program = np.array(program)
    signal = 0
    for phase in phases:
        signal = IntcodeVM(program).run([phase, signal])
    assert signal == expected_output

def test_day7_part1():
    program = get_program(day=7)
    phases = (3, 0, 4, 2, 1)
    signal = 0
    for phase in phases:
        signal = IntcodeVM(program).run([phase, signal])
    assert signal == 30940

def exec_day7_part2(phases, program):
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

# fmt: off
@pytest.mark.parametrize(["program", "phases", "expected_output"], [
    ((3,26,1001,26,-4,26,3,27,1002,27,2,27,1,27,26,27,4,27,1001,28,-1,28,1005,28,6,99,0,0,5), (9,8,7,6,5), 139629729),
    ((3,52,1001,52,-5,52,3,53,1,52,56,54,1007,54,5,55,1005,55,26,1001,54,-5,54,1105,1,12,1,53,54,53,1008,54,0,55,1001,55,1,55,2,53,55,53,4,53,1001,56,-1,56,1005,56,6,99,0,0,0,0,10), (9,7,8,5,6), 18216)
])
# fmt: on
def test_day7_part2_examples(program, phases, expected_output):
    assert exec_day7_part2(phases, program) == expected_output

    
def test_day7_part2():
    program = get_program(day=7)
    phases = (8, 9, 6, 7, 5)
    assert exec_day7_part2(phases, program) == 76211147
    
    
# fmt: off
@pytest.mark.parametrize(["program", "expected_output"], [
    ((109,1,204,-1,1001,100,1,100,1008,100,16,101,1006,101,0,99), [109,1,204,-1,1001,100,1,100,1008,100,16,101,1006,101,0,99]),
    ((1102,34915192,34915192,7,4,7,99,0), 1219070632396864),
    ((104,1125899906842624,99), 1125899906842624)
])
# fmt: on
def test_day9_part1_examples(program, expected_output):
    assert IntcodeVM(program).run() == expected_output

def test_day9_part1():
    program = get_program(day=9)
    assert IntcodeVM(program).run(1) == 3409270027

def test_day9_part2():
    program = get_program(day=9)
    assert IntcodeVM(program).run(2) == 82760

def test_day11_part1():
    program = get_program(day=11)

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
    assert len(tiles) == 2336

def test_day11_part2():
    program = get_program(day=11)
    
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

    output = "\n".join("".join(map(pprint, line)) for line in result)
    lines = output.split("\n")
    assert lines[0].strip() == "█  █ ████  ██  ████ █  █ ███  █    ███".strip()
    assert lines[1].strip() == "█  █    █ █  █ █    █ █  █  █ █    █  █".strip()
    assert lines[2].strip() == "█  █   █  █  █ ███  ██   ███  █    █  █".strip()
    assert lines[3].strip() == "█  █  █   ████ █    █ █  █  █ █    ███".strip()
    assert lines[4].strip() == "█  █ █    █  █ █    █ █  █  █ █    █".strip()
    assert lines[5].strip() == " ██  ████ █  █ ████ █  █ ███  ████ █".strip()

def test_day13_part1():
    program = get_program(day=13)
    outputs = IntcodeVM(program).run()
    x, y, tile = outputs[::3], outputs[1::3], outputs[2::3]
    points = [(xx, yy) for xx, yy, t in zip(x, y, tile) if t == 2]
    assert len(points) == 284


class Day13Game:
    def __init__(self):
        self.world = None
        self.ball_pos = None
        self.paddle_pos = None
        self.score = 0
        self.char_map = {
            0: " ",
            1: "█",
            2: "#",
            3: "=",
            4: "O"
        }
    
    def update(self, outputs):
        xs, ys, tiles = np.array(outputs[::3]), np.array(outputs[1::3]), np.array(outputs[2::3])
        
        if self.world is None:
            self.world = np.zeros(shape=(ys.max() + 1, xs.max() + 1), dtype=str)
        
        for x, y, t in zip(xs, ys, tiles):
            if t == 4:
                self.ball_pos = (x, y)
            elif t == 3:
                self.paddle_pos = (x, y)

            if x == -1:
                self.score = t
            else:
                self.world[y, x] = self.char_map[t]
    
    def visualize(self):
        print(f"Score: {self.score}")
        print("\n".join("".join(line) for line in self.world))


def test_day13_part2():
    program = get_program(day=13)
    program[0] = 2
    vm = IntcodeVM(program)
    outputs = vm.run()
    game = Day13Game()
    game.update(outputs)

    while not vm.stopped:
        direction = np.sign(game.ball_pos[0] - game.paddle_pos[0])
        outputs = vm.run(direction)
        game.update(outputs)
    
    assert game.score == 13581
