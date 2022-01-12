"""
The class Segment. A segment contains a list of paths.
"""

from typing import List, Set, Tuple

from path import Path

Point = Tuple[int, int]


class Segment:
    """
    Represents one segment in a state, and the paths are ordered.
    """

    def __init__(self, paths: List[Path]) -> None:
        self.paths = paths
        assert self.consistency_check(full=False)

    def consistency_check(self, full: bool = False) -> bool:
        """
        verifies that
         - each segment has at least one path
         - the paths are disjoint
        """
        assert sum(len(path) for path in self.paths) == len(self.points()), repr(self)
        if full:
            assert all(path.consistency_check() for path in self.paths)
        return True

    def points(self) -> Set[Point]:
        """
        returns the set of points contained in the paths in this segment
        """
        if len(self.paths) == 0:
            return set()
        return set.union(*[set(path.points) for path in self.paths])

    def contains_edge(self, point1: Point, point2: Point) -> bool:
        """
        returns True if point1 and point2 occur consecutively in the path,
        and in that order
        """
        return any(path.contains_edge(point1, point2) for path in self.paths)

    def trim(self) -> "Segment":
        """
        Shifts all paths in the segment left by 1 unit. This may cause a path to break
        into several paths (which all remain in this segment).
        """
        return Segment(
            [path_portion for path in self.paths for path_portion in path.trim()]
        )

    def find_path(self, point: Point) -> Path:
        """
        Returns the path in this segment that contains a particular point.
        """
        for path in self.paths:
            if point in path.points:
                return path
        assert False

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Segment):
            return NotImplemented
        return self.paths == other.paths

    def __hash__(self) -> int:
        return hash(tuple(self.paths))

    def __contains__(self, point: Point) -> bool:
        return any(point in path for path in self.paths)

    def __str__(self) -> str:
        return str(self.paths)

    def __repr__(self) -> str:
        return f"Segment({repr(self.paths)})"
