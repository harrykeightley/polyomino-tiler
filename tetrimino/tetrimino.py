import operator
import queue
import random
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from functools import cached_property, reduce
from itertools import chain, permutations, product, repeat
from typing import Callable, Iterable, Optional, Set, Tuple, TypeVar

import numpy as np
from numpy._typing import NDArray

Dimensions = Tuple[int, int]
Position = Tuple[int, int]


@dataclass
class GridShape:
    pattern: list[list[bool]]

    def set(self, value: bool, position: Position):
        row, col = position
        self.pattern[row][col] = value

    @property
    def np(self):
        return np.array(self.pattern, dtype=bool)

    @classmethod
    def from_np(cls, arr: NDArray):
        return cls(arr.tolist())

    def map_np(self, fn: Callable[[NDArray], NDArray]):
        return GridShape.from_np(fn(self.np))

    def shape_fits(self, shape: "GridShape") -> bool:
        return all(map(operator.ge, self.np.shape, shape.np.shape))

    def add_rows(self, rows: int, prepend=True) -> "GridShape":
        row_size = self.np.shape[1]
        empty_row = [False for _ in range(row_size)]

        if prepend:
            new_rows = *repeat(empty_row, rows), *self.np
        else:
            new_rows = *self.np, *repeat(empty_row, rows)

        return GridShape.from_np(np.row_stack(new_rows))

    def add_cols(self, cols: int, prepend=True) -> "GridShape":
        return self.map_np(np.transpose).add_rows(cols, prepend).map_np(np.transpose)

    def add_offset(self, offset: Position) -> "GridShape":
        rows, cols = offset
        return self.add_rows(rows).add_cols(cols)

    def resize_to(self, target_shape: "GridShape") -> "GridShape":
        # Assumes target_shape can fit the current one
        row_delta, col_delta = map(operator.sub, target_shape.np.shape, self.np.shape)

        return self.add_rows(row_delta, prepend=False).add_cols(
            col_delta, prepend=False
        )

    def crop_to(self, target_shape: "GridShape") -> "GridShape":
        rows, cols = target_shape.np.shape
        return GridShape.from_np(self.np[:rows, :cols])

    def fits(self, shape: "GridShape", offset: Position = (0, 0)) -> bool:
        shape = shape.add_offset(offset)
        if not self.shape_fits(shape):
            return False

        shape = shape.resize_to(self)
        # shapes are now same dimensions

        return not any(np.multiply(self.np, shape.np).flatten())

    def mask(self, shape: "GridShape", offset: Position = (0, 0)) -> "GridShape":
        shape = shape.add_offset(offset).crop_to(self).resize_to(self)
        return GridShape.from_np((self.np + shape.np).astype(bool))

    def display(self, letter: str):
        char = lambda index: letter if index else "_"
        make_row = lambda row: "".join(map(char, row))
        return "\n".join(map(make_row, self.pattern))

    def possible_offsets(self, shape: "GridShape") -> Iterable[Position]:
        if not self.shape_fits(shape):
            return []

        row_delta, col_delta = map(operator.sub, self.np.shape, shape.np.shape)
        return map(tuple, product(range(row_delta + 1), range(col_delta + 1)))  # type: ignore

    def count_free_spaces(self) -> int:
        return len(list(filter(lambda x: not x, self.np.flatten().astype(bool))))

    def count_taken_spaces(self) -> int:
        rows, cols = self.np.shape
        return rows * cols - self.count_free_spaces()

    def bounding_free_area(self):
        if self.count_free_spaces() == 0:
            return 0

        a = np.where(self.np == False)
        dx = np.max(a[1]) - np.min(a[1]) + 1
        dy = np.max(a[0]) - np.min(a[0]) + 1
        return dx * dy

    def __str__(self) -> str:
        return self.display("X")

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, self.__class__):
            return False

        shape = __value
        return shape.np == self.np

    def __hash__(self) -> int:
        return hash(str(self.np))


@dataclass
class Tetrimino(GridShape):
    letter: str

    def rotate(self, clockwise=True) -> "Tetrimino":
        flip = np.fliplr if clockwise else np.flipud
        transpose = np.array(self.pattern).transpose()
        return Tetrimino(flip(transpose).tolist(), self.letter)

    def rotate_n(self, n: int) -> "Tetrimino":
        result = self
        for _ in range(n % 4):
            result = result.rotate()
        return result

    def rotations(self) -> list["Tetrimino"]:
        return [self, self.rotate(), self.rotate().rotate(), self.rotate(False)]

    def mirror(self, vertical=True) -> "Tetrimino":
        flip = np.fliplr if vertical else np.flipud
        pattern = flip(np.array(self.pattern)).tolist()
        return Tetrimino(pattern, self.letter)

    def __str__(self) -> str:
        return self.display(self.letter)

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, self.__class__):
            return False

        t = __value
        return t.letter == self.letter

    def __hash__(self) -> int:
        return hash(self.letter)


class Tetriminos(Enum):
    I = Tetrimino([[True, True, True, True]], "I")
    O = Tetrimino([[True, True], [True, True]], "O")
    T = Tetrimino([[False, True, False], [True, True, True]], "T")
    S = Tetrimino([[False, True, True], [True, True, False]], "S")
    Z = Tetrimino([[True, True, False], [False, True, True]], "Z")
    L = Tetrimino([[True, True, True], [True, False, False]], "L")
    J = Tetrimino([[True, True, True], [False, False, True]], "J")


def create_empty_grid(dimensions: Dimensions) -> GridShape:
    return GridShape.from_np(np.zeros(dimensions, dtype=bool))


@dataclass
class Piece:
    tetrimino: Tetrimino
    rotation: int = 0
    offset: Position = (0, 0)

    @cached_property
    def shape(self) -> GridShape:
        return self.tetrimino.rotate_n(self.rotation).add_offset(self.offset)

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, self.__class__):
            return False
        piece = __value
        return (
            piece.offset == self.offset
            and piece.rotation == self.rotation
            and piece.tetrimino == self.tetrimino
        )

    def __hash__(self) -> int:
        return hash((self.tetrimino, self.rotation, self.offset))


T = TypeVar("T")


def search(
    initial: T,
    neighbours: Callable[[T], Iterable[T]],
    cost: Callable[[T], int],
    heuristic: Callable[[T], int],
    is_finished: Callable[[T], bool],
):
    def value(node: T) -> Tuple[int, T]:
        return (cost(node) + heuristic(node), node)

    q = queue.PriorityQueue[Tuple[int, T]]()
    q.put(value(initial))

    seen: Set[T] = set()
    seen.add(initial)

    while not q.empty():
        _, node = q.get()
        print()
        print(node)
        if is_finished(node):
            return node

        for neighbour in neighbours(node):
            if neighbour not in seen:
                q.put(value(neighbour))

            seen.add(neighbour)

    return None


@dataclass
class State:
    grid: GridShape
    pieces: list[Piece]
    action_space: list[Piece]

    def cost(self):
        taken_squares = sum(piece.shape.count_taken_spaces() for piece in self.pieces)
        return taken_squares

    def piece_counts(self) -> dict[str, int]:
        return Counter(piece.tetrimino.letter for piece in self.pieces)

    def concentration(self) -> int:
        return sum(map(lambda x: x**2, self.piece_counts().values()))

    def place(self, piece: Piece, action_space: list[Piece]) -> "State":
        return State(
            self.grid.mask(piece.shape), self.pieces + [piece], action_space.copy()
        )

    def neighbours(self):
        valid_pieces = list(
            filter(lambda piece: self.grid.fits(piece.shape), self.action_space)
        )
        result = list(
            map(lambda piece: self.place(piece, list(valid_pieces)), valid_pieces)
        )
        random.shuffle(result)
        return result

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, self.__class__):
            return False
        state = __value
        return self.pieces == state.pieces

    def __hash__(self) -> int:
        return hash((self.grid, *self.pieces))

    def __lt__(self, other: "State") -> bool:
        return len(self.pieces) < len(other.pieces)

    def __str__(self) -> str:
        if len(self.pieces) == 0:
            return str(self.grid)

        merge_char = lambda a, b: a if a != "_" else b
        merge_strings = lambda a, b: "".join(map(merge_char, a, b))
        return reduce(
            merge_strings,
            (
                piece.shape.resize_to(self.grid).display(piece.tetrimino.letter)
                for piece in self.pieces
            ),
        )


# if action space wasn't valid last time, wont be valid next time


def generate_action_space(grid: GridShape, pieces: list[Tetrimino]):
    ROTATED_PIECES = product(pieces, range(4))

    result: list[Piece] = []
    for t, rotation in ROTATED_PIECES:
        for offset in grid.possible_offsets(t.rotate_n(rotation)):
            result.append(Piece(t, rotation, offset))

    return result


def generate_solution(shape: GridShape, pieces: list[Tetrimino]) -> Optional[State]:
    action_space = generate_action_space(shape, pieces)
    # for piece in action_space:
    #     print("\n" + str(piece.shape))
    initial = State(shape, [], action_space)

    def heuristic(state: State) -> int:
        return state.grid.count_free_spaces() + state.grid.bounding_free_area()

    def is_finished(state: State) -> bool:
        return state.grid.count_free_spaces() <= shape.count_free_spaces() % 4

    return search(initial, State.neighbours, State.cost, heuristic, is_finished)


# solution = generate_solution(create_empty_grid((6, 6)), [t.value for t in Tetriminos])
# print("Found one")
# print(solution)
grid = create_empty_grid((3, 2))
grid.bounding_free_area()
grid = grid.mask(Tetriminos.O.value)
grid = grid.mask(Tetriminos.O.value, (0, 2))
print(grid)
grid = grid.mask(Tetriminos.I.value.rotate(), (2, 0))


def solve(shape: GridShape, pieces: list[Tetrimino]) -> Optional[State]:
    ...
