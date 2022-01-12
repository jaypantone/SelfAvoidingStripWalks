"""
The class Path representing one path in a state.
"""

from typing import List, Tuple

Point = Tuple[int, int]


class Path:
    """
    Represents one path in a state.
    """

    def __init__(self, points: List[Point]) -> None:
        """
        <points> is the (ordered) list of points in the path
        <allow_col_2> allows the path to have points in the third column (x-value 2)
           which is useful temporarily when building new states
        """
        self.points = points
        assert self.consistency_check()

    def consistency_check(self) -> bool:
        """
        verifies that
         - each path has at least one point
         - the points are distinct
         - the points all have x-value 0 or 1 and y-value that is nonnegative
         - each point connects to the next
        """
        assert len(self.points) > 0
        assert len(self.points) == len(set(self.points))
        assert all(pt[0] >= 0 and pt[1] >= 0 for pt in self.points)
        for index in range(len(self.points) - 1):
            pt1 = self.points[index]
            pt2 = self.points[index + 1]
            assert (pt1[0] == pt2[0] and pt1[1] in {pt2[1] - 1, pt2[1] + 1}) or (
                pt1[0] in {pt2[0] - 1, pt2[0] + 1} and pt1[1] == pt2[1]
            )
        return True

    def contains_edge(self, point1: Point, point2: Point) -> bool:
        """
        returns True if point1 and point2 occur consecutively in the path,
        and in that order
        """
        return any(
            self.points[i] == point1 and self.points[i + 1] == point2
            for i in range(len(self) - 1)
        )

    def endpoints(self) -> Tuple[Point, Point]:
        """
        Returns the first and last point in the path. If the path consists of only one
        point, then it returns it twice.
        """
        return (self.points[0], self.points[-1])

    def trim(self) -> List["Path"]:
        """
        Shifts all points in the path left by 1 unit. This may cause a path to break
        into several paths, which is why the return type is a list of paths.
        """
        new_paths = []
        cur_path = []
        for point in self.points:
            if point[0] > 0:
                cur_path.append((point[0] - 1, point[1]))
            else:
                if len(cur_path) > 0:
                    new_paths.append(Path(cur_path))
                    cur_path = []
        if len(cur_path) > 0:
            new_paths.append(Path(cur_path))
        return new_paths

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Path):
            return NotImplemented
        return self.points == other.points

    def __hash__(self) -> int:
        return hash(tuple(self.points))

    def __len__(self) -> int:
        return len(self.points)

    def __contains__(self, point: Point) -> bool:
        return point in self.points

    def __str__(self) -> str:
        return str(self.points)

    def __repr__(self) -> str:
        return f"Path({repr(self.points)})"
