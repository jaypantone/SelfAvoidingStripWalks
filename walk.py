"""
Provides the Walk class and functions to count the number of walks with certain
dimensions with various maximality conditions by brute force. This is for testing
and confirmation of the finite state machines built by the other classes.
"""
from collections import deque
from copy import copy
from fractions import Fraction
from typing import Deque, Dict, List, Optional, Set, Tuple, Union

import sympy  # type: ignore

Point = Tuple[int, int]

Weight = Union[sympy.polys.polytools.Poly, sympy.Expr]


C = sympy.symbols("C")


class Walk:
    """
    The class Walk represents a walk in a rectangle of fixed height and width. All walks
    start in the top-left corner, but they do not have to be maximal because this class
    will be used to represent the partial walks as they are built.
    """

    def __init__(
        self,
        height: int,
        width: int,
        walk: Optional[List[Point]] = None,
        occupied_points: Optional[Set[Point]] = None,
    ):
        self.height = height
        self.width = width
        if walk is None:
            self.walk = [(0, height - 1)]
        else:
            self.walk = walk
        if occupied_points is None:
            self.occupied_points = set(walk)
        else:
            self.occupied_points = occupied_points
            assert self.occupied_points == set(walk)

    def get_next_walks(self) -> List["Walk"]:
        """
        move the walk forward in all possible ways
        """
        end = self.walk[-1]
        next_points: List[Point] = [None] * 4
        if end[0] > 0:
            next_points[0] = (end[0] - 1, end[1])
        if end[0] < self.width - 1:
            next_points[1] = (end[0] + 1, end[1])
        if end[1] > 0:
            next_points[2] = (end[0], end[1] - 1)
        if end[1] < self.height - 1:
            next_points[3] = (end[0], end[1] + 1)

        to_return: List[Walk] = []
        for np in next_points:
            if np is not None and np not in self.occupied_points:
                new_points = copy(self.occupied_points)
                new_points.add(np)
                to_return.append(
                    Walk(self.height, self.width, self.walk + [np], new_points)
                )
        return to_return

    def get_next_probabilistic_walks(
        self, weight: Fraction
    ) -> List[Tuple["Walk", Fraction]]:
        """
        move the walk forward in all possible ways, and return each new walk
        along with the fraction 1 / [number of open locations that it could have moved]
        """
        end = self.walk[-1]
        next_points: List[Point] = [None] * 4
        if end[0] > 0:
            next_points[0] = (end[0] - 1, end[1])
        if end[0] < self.width - 1:
            next_points[1] = (end[0] + 1, end[1])
        if end[1] > 0:
            next_points[2] = (end[0], end[1] - 1)
        if end[1] < self.height - 1:
            next_points[3] = (end[0], end[1] + 1)

        open_neighbors = [
            np
            for np in next_points
            if np is not None and np not in self.occupied_points
        ]
        if len(open_neighbors) == 0:
            return []

        frac = Fraction(1, len(open_neighbors))

        to_return: List[Tuple[Walk, Fraction]] = []
        for np in open_neighbors:
            new_points = copy(self.occupied_points)
            new_points.add(np)
            to_return.append(
                (
                    Walk(self.height, self.width, self.walk + [np], new_points),
                    weight * frac,
                )
            )
        return to_return

    def get_next_energistic_walks(self, weight: Weight) -> List[Tuple["Walk", Weight]]:
        """
        move the walk forward in all possible ways, and return each new walk
        along with the energistic probability
        """
        end = self.walk[-1]
        next_points: List[Point] = [None] * 4
        if end[0] > 0:
            next_points[0] = (end[0] - 1, end[1])
        if end[0] < self.width - 1:
            next_points[1] = (end[0] + 1, end[1])
        if end[1] > 0:
            next_points[2] = (end[0], end[1] - 1)
        if end[1] < self.height - 1:
            next_points[3] = (end[0], end[1] + 1)

        def neighbors(pt: Point) -> List[Point]:
            return [
                (pt[0] - 1, pt[1]),
                (pt[0] + 1, pt[1]),
                (pt[0], pt[1] - 1),
                (pt[0], pt[1] + 1),
            ]

        energy: Dict[Point, Weight] = {
            pt: C ** len([np for np in neighbors(pt) if np in self.occupied_points])
            for pt in next_points
            if pt is not None and pt not in self.occupied_points
        }
        total_energy = sum(energy.values())

        if len(energy) == 0:
            return []

        to_return: List[Tuple[Walk, Fraction]] = []
        for np in energy.keys():
            new_points = copy(self.occupied_points)
            new_points.add(np)
            to_return.append(
                (
                    Walk(self.height, self.width, self.walk + [np], new_points),
                    weight * ((energy[np] / C) / (total_energy / C)),
                )
            )
        return to_return

    def __len__(self) -> int:
        return len(self.walk) - 1

    def __repr__(self) -> str:
        return f"Walk(height={self.height}, width={self.width}, walk={self.walk})"

    def __str__(self) -> str:
        width = self.width
        height = self.height
        walk_pairs = set(zip(self.walk[:-1], self.walk[1:]))

        start_color = "\x1b[42m"
        end_color = "\x1b[0m"

        S = "-" * (5 * width) + "\n"
        for y_val in range(height - 1, -1, -1):
            S += "| "

            # first do the horizontal row of points
            for x_val in range(width - 1):
                if (x_val, y_val) in self.walk:
                    S += f"{start_color}*{end_color}"
                else:
                    S += "*"
                if ((x_val, y_val), (x_val + 1, y_val)) in walk_pairs:
                    S += f"{start_color} >> {end_color}"
                elif ((x_val + 1, y_val), (x_val, y_val)) in walk_pairs:
                    S += f"{start_color} << {end_color}"
                else:
                    S += "    "
            if (width - 1, y_val) in self.walk:
                S += f"{start_color}*{end_color}"
            else:
                S += "*"
            S += " |"

            # now we do the arrows below it, but not when y_val = 0
            if y_val > 0:
                S += "\n| "
                ver_arrows = []
                for x_val in range(width):
                    if ((x_val, y_val - 1), (x_val, y_val)) in walk_pairs:
                        ver_arrows.append(f"{start_color}^{end_color}")
                    elif ((x_val, y_val), (x_val, y_val - 1)) in walk_pairs:
                        ver_arrows.append(f"{start_color}v{end_color}")
                    else:
                        ver_arrows.append(" ")
                S += "    ".join(ver_arrows)
                S += " |\n| "
                S += "    ".join(ver_arrows)
                S += " |\n"

        S += "\n" + "-" * (5 * width)

        return S


### A walk is *maximal* if it cannot take any more steps.
### A walk is *anchored* if it starts in the top-left corner.
### All of our walks are directed by default.


def count_directed_non_maximal_unanchored_walks(height: int, width: int) -> int:
    """
    Returns the number of directed non-maximal unanchored walks in a <height> x <width>
    rectangle. Does not count the walks by length. To count undirected walks, divide
    the output by 2.
    """
    enum = 0
    todo: List[Walk] = [
        Walk(height, width, [(x_val, y_val)])
        for x_val in range(width)
        for y_val in range(height)
    ]
    while todo:
        enum += 1
        walk = todo.pop()
        # Walks can only ever be made in one way, so we don't have to worry about adding
        #   work to <todo> that has already been done.
        todo.extend(walk.get_next_walks())
    return enum


def count_directed_non_maximal_unanchored_walks_by_length(
    height: int, width: int, max_len: Optional[int] = None
) -> List[int]:
    """
    Returns the number of directed non-maximal unanchored walks in a <height> x <width>
    rectangle, refined by length. To count undirected walks, divide the output by 2.
    """
    enum = [0] * (max_len + 1 if max_len else height * width)
    print(len(enum), enum)
    todo: Deque[Walk] = deque(
        [
            Walk(height, width, [(x_val, y_val)])
            for x_val in range(width)
            for y_val in range(height)
        ]
    )
    while todo:
        walk = todo.popleft()
        walk_len = len(walk)
        enum[walk_len] += 1
        # Walks can only ever be made in one way, so we don't have to worry about adding
        #   work to <todo> that has already been done.
        if not max_len or walk_len < max_len:
            todo.extend(walk.get_next_walks())
    return enum


def count_non_maximal_anchored_walks(height: int, width: int) -> int:
    """
    Returns the number of non-maximal anchored walks in a <height> x <width> rectangle.
    Does not count the walks by length.
    """
    enum = 0
    todo: List[Walk] = [Walk(height, width, [(0, height - 1)])]
    while todo:
        enum += 1
        walk = todo.pop()
        # Walks can only ever be made in one way, so we don't have to worry about adding
        #   work to <todo> that has already been done.
        todo.extend(walk.get_next_walks())
    return enum


def count_directed_non_maximal_anchored_walks_by_length(
    height: int, width: int, max_len: Optional[int] = None
) -> List[int]:
    """
    Returns the number of directed non-maximal anchored walks in a <height> x <width>
    rectangle, refined by length.
    """
    enum = [0] * (max_len + 1 if max_len else height * width)
    print(len(enum), enum)
    todo: Deque[Walk] = deque([Walk(height, width, [(0, height - 1)])])
    while todo:
        walk = todo.popleft()
        walk_len = len(walk)
        enum[walk_len] += 1
        # Walks can only ever be made in one way, so we don't have to worry about adding
        #   work to <todo> that has already been done.
        if not max_len or walk_len < max_len:
            todo.extend(walk.get_next_walks())
    return enum


def count_maximal_anchored_walks(height: int, width: int) -> int:
    """
    Returns the number of maximal anchored walks in a <height> x <width> rectangle.
    Does not count the walks by length.
    """
    enum = 0
    todo: List[Walk] = [Walk(height, width, [(0, height - 1)])]
    while todo:
        walk = todo.pop()
        next_walks = walk.get_next_walks()
        if next_walks:
            # Walks can only ever be made in one way, so we don't have to worry about adding
            #   work to <todo> that has already been done.
            todo.extend(next_walks)
        else:
            enum += 1
    return enum


def count_maximal_anchored_walks_by_length(
    height: int, width: int, max_len: Optional[int] = None
) -> List[int]:
    """
    Returns the number of maximal anchored walks in a <height> x <width> rectangle,
    refined by length.
    """
    enum = [0] * (max_len + 1 if max_len else height * width)
    # print(len(enum), enum)
    todo: Deque[Walk] = deque([Walk(height, width, [(0, height - 1)])])
    while todo:
        walk = todo.popleft()
        walk_len = len(walk)
        # Walks can only ever be made in one way, so we don't have to worry about adding
        #   work to <todo> that has already been done.
        next_walks = walk.get_next_walks()
        if next_walks:
            if not max_len or walk_len < max_len:
                todo.extend(next_walks)
        else:
            enum[walk_len] += 1

    return enum


def count_maximal_anchored_full_walks(height: int, width: int) -> int:
    """
    Returns the number of maximal anchored walks in a <height> x <width> rectangle,
    refined by length.
    """
    enum = 0
    # print(len(enum), enum)
    todo: Deque[Walk] = deque([Walk(height, width, [(0, height - 1)])])
    while todo:
        walk = todo.popleft()
        walk_len = len(walk)
        # Walks can only ever be made in one way, so we don't have to worry about adding
        #   work to <todo> that has already been done.
        next_walks = walk.get_next_walks()
        if next_walks:
            todo.extend(next_walks)
        elif walk_len == height * width - 1:
            enum += 1

    return enum


def count_probabilistic_maximal_anchored_walks_by_length(
    height: int, width: int, max_len: Optional[int] = None
) -> List[Fraction]:
    """
    Returns the number of maximal anchored walks in a <height> x <width> rectangle,
    refined by length.
    """
    enum: List[Fraction] = [
        Fraction(0, 1) for i in range(max_len + 1 if max_len else height * width)
    ]
    # print(len(enum), enum)
    todo: Deque[Tuple[Walk, Fraction]] = deque(
        [(Walk(height, width, [(0, height - 1)]), Fraction(1, 1))]
    )
    while todo:
        walk, weight = todo.popleft()
        walk_len = len(walk)
        # Walks can only ever be made in one way, so we don't have to worry about adding
        #   work to <todo> that has already been done.
        next_walks = walk.get_next_probabilistic_walks(weight)
        if next_walks:
            if not max_len or walk_len < max_len:
                todo.extend(next_walks)
        else:
            enum[walk_len] += weight

    return enum


def count_energistic_maximal_anchored_walks_by_length(
    height: int, width: int, max_len: Optional[int] = None
) -> List[Weight]:
    """
    Returns the number of maximal anchored walks in a <height> x <width> rectangle,
    refined by length.
    """
    enum = [
        sympy.simplify(0) for i in range(max_len + 1 if max_len else height * width)
    ]
    # print(len(enum), enum)
    todo: Deque[Tuple[Walk, Weight]] = deque(
        [(Walk(height, width, [(0, height - 1)]), sympy.sympify(1))]
    )
    while todo:
        walk, weight = todo.popleft()
        walk_len = len(walk)
        # Walks can only ever be made in one way, so we don't have to worry about adding
        #   work to <todo> that has already been done.
        next_walks = walk.get_next_energistic_walks(weight)
        if next_walks:
            if not max_len or walk_len < max_len:
                todo.extend(next_walks)
        else:
            enum[walk_len] += weight

    return [term.factor() for term in enum]


def get_maximal_anchored_walks(height: int, width: int) -> List[Walk]:
    """
    Returns a list of the maximal anchored walks in a <height> x <width> rectangle.
    Does not count the walks by length.
    """
    todo: List[Walk] = [Walk(height, width, [(0, height - 1)])]
    done: List[Walk] = []
    while todo:
        walk = todo.pop()
        next_walks = walk.get_next_walks()
        if next_walks:
            # Walks can only ever be made in one way, so we don't have to worry about adding
            #   work to <todo> that has already been done.
            todo.extend(next_walks)
        else:
            done.append(walk)
    return done
