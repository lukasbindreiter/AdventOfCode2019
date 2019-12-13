"""
Implementation of an Intcode Virtual machine for advent of code 2019

Basic usage
>>> program = np.array([1, 2, 3, 4, ...])
>>> inp = 3
>>> output = IntcodeVM(program).run(inp)
>>> print(output)
"""

from dataclasses import dataclass
from typing import Callable, Union, List
from functools import partial
import numpy as np


class ProgramStop(Exception):
    pass


class InputRequired(Exception):
    pass


@dataclass
class OpCode:
    name: str
    code: int
    n_args: int
    func: Callable


def exit_(computer):
    raise ProgramStop()


def arithmetic(computer, op):
    computer.c = op(computer.a, computer.b)


def input_(computer):
    if len(computer.inputs) == 0:
        raise InputRequired()
    computer.a = computer.inputs.pop(0)


def output(computer):
    computer.outputs.append(computer.a)


def jump_if(computer, condition):
    if condition(computer.a):
        return computer.b


def arithmethic_comparison(computer, condition):
    computer.c = int(condition(computer.a, computer.b))


def offset_relative_base(computer):
    computer._relative_base += computer.a


code_map = {
    op.code: op
    for op in [
        OpCode(
            name="add",
            code=1,
            n_args=3,
            func=partial(arithmetic, op=lambda a, b: a + b),
        ),
        OpCode(
            name="multiply",
            code=2,
            n_args=3,
            func=partial(arithmetic, op=lambda a, b: a * b),
        ),
        OpCode(name="input", code=3, n_args=1, func=input_),
        OpCode(name="output", code=4, n_args=1, func=output),
        OpCode(
            name="jump-if-true",
            code=5,
            n_args=2,
            func=partial(jump_if, condition=lambda a: a != 0),
        ),
        OpCode(
            name="jump-if-false",
            code=6,
            n_args=2,
            func=partial(jump_if, condition=lambda a: a == 0),
        ),
        OpCode(
            name="less than",
            code=7,
            n_args=3,
            func=partial(arithmethic_comparison, condition=lambda a, b: a < b),
        ),
        OpCode(
            name="equals",
            code=8,
            n_args=3,
            func=partial(arithmethic_comparison, condition=lambda a, b: a == b),
        ),
        OpCode(name="offset-relbase", code=9, n_args=1, func=offset_relative_base),
        OpCode(name="exit", code=99, n_args=0, func=exit_),
    ]
}


def parse_opcode(code: int):
    """
    Parse the opcode and the parameter modes from the given opcode integer
    
    >>> parse_opcode(1002)
    OpCode('multiplay'), [0, 1, 0]
    
    >>> parse_opcode(1107)
    OpCode('less than'), [1, 1, 0]
    """
    instruction = code_map[code % 100]
    modes = [code // 10 ** (p + 2) % 10 for p in range(instruction.n_args)]
    return instruction, modes


class IntcodeVM:
    def __init__(self, memory, debug=False):
        self.memory = np.asarray(memory).copy()
        self.inputs = []
        self.outputs = []
        self.debug = debug
        self.stopped = False

        # Program state:
        self._ip = 0
        self._relative_base = 0

        self._instruction = None
        self._param_modes = None
        self._params = None

    def run(self, inputs: Union[int, List[int], None] = None):
        """
        Run the Intcode VM until the program terminates or an input is required that is not yet available
        """
        if inputs is None:
            inputs = []
        try:
            self.inputs.extend(inputs)
        except TypeError:  # single integer
            self.inputs.append(inputs)

        if self.stopped:
            raise ProgramStop("Program already terminated, can't run anymore")

        self.outputs = []

        try:
            while self._ip < len(self.memory):
                self.step()
        except ProgramStop:
            self.stopped = True
        except InputRequired:
            pass

        return self.outputs[0] if len(self.outputs) == 1 else self.outputs

    def step(self):
        """
        Execute the current instruction
        """
        self._instruction, self._param_modes = parse_opcode(self.memory[self._ip])
        self._params = self.memory[
            self._ip + 1 : self._ip + 1 + self._instruction.n_args
        ]
        self._log_instruction_call_before()
        new_ip = self._instruction.func(self)
        self._log_instruction_call_after(new_ip)
        if new_ip is None:
            self._ip += 1 + self._instruction.n_args
        else:
            self._ip = new_ip

    def _ensure_enough_memory(self, n):
        while len(self.memory) < n:
            new_mem = np.zeros(len(self.memory) * 2, np.int)
            new_mem[: len(self.memory)] = self.memory
            self.memory = new_mem

    # easy accessors for the parameters of the current instruction
    def _get_param(self, index):
        assert len(self._params) > index
        if self._param_modes[index] == 0:
            self._ensure_enough_memory(self._params[index])
            return self.memory[self._params[index]]
        elif self._param_modes[index] == 1:
            return self._params[index]
        else:  # mode 2
            self._ensure_enough_memory(self._relative_base + self._params[index])
            return self.memory[self._relative_base + self._params[index]]

    def _set_param(self, index, value):
        assert len(self._params) > index
        if self._param_modes[index] == 0:
            self._ensure_enough_memory(self._params[index])
            self.memory[self._params[index]] = value
        elif self._param_modes[index] == 2:
            self._ensure_enough_memory(self._relative_base + self._params[index])
            self.memory[self._relative_base + self._params[index]] = value
        else:
            raise ValueError(
                f"Wrong parameter mode for writing to a value: {self._param_modes[index]}"
            )

    @property
    def a(self):
        return self._get_param(0)

    @a.setter
    def a(self, val):
        self._set_param(0, val)

    @property
    def b(self):
        return self._get_param(1)

    @b.setter
    def b(self, val):
        self._set_param(1, val)

    @property
    def c(self):
        return self._get_param(2)

    @c.setter
    def c(self, val):
        self._set_param(2, val)

    def _log_instruction_call_before(self):
        if not self.debug:
            return
        instr = " ".join(
            map(str, self.memory[self._ip : self._ip + 1 + self._instruction.n_args])
        )
        params = [
            f"[{par}]={self.memory[par]}" if m == 0 else str(par)
            for par, m in zip(self._params, self._param_modes)
        ]
        print(
            f"{str(self._ip) + '  ':.<6} {instr+'  ':.<20}  {self._instruction.name+'  ':.<15}  {', '.join(params) + '  ':.<40}",
            end="",
        )

    def _log_instruction_call_after(self, new_ip):
        if not self.debug:
            return
        writeable_params = set(
            [par for par, m in zip(self._params, self._param_modes) if m == 0]
        )
        values = [f"mem[{par}] = {self.memory[par]}" for par in writeable_params]
        if new_ip:
            print(f"✓ Instruction Pointer changed: New value {new_ip}")
        elif values:
            print(f"✓ Result: {', '.join(values)}")
        else:
            print("✓ No effect to memory or execution")


def parse_program(input_data: str) -> np.array:
    return np.array(list(map(int, input_data.strip().split(","))))
