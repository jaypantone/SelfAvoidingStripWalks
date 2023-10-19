"""
The class State representing a state in the finite state machine.
A State is a list of Segments. A Segment is a list of Paths.
"""

from copy import copy, deepcopy
from fractions import Fraction
from typing import Dict, List, Set, Tuple, Union

import sympy  # type: ignore

from path import Path
from segment import Segment

Point = Tuple[int, int]
# Tag = str
Weight = Union[sympy.polys.polytools.Poly, sympy.Expr]

x, C = sympy.symbols("x C")


class State:
    """
    New version of a state where we just always assume we are building on the
    infinite half grid graph, so we don't need to separate final and non-final
    states. But, we will use the "final" flag to mark a state as accepting for
    convenience.
    """

    def __init__(
        self,
        height: int,
        width: int,
        segments: List[Segment],
        real_cols: int,
        final: bool,
    ) -> None:
        self.height = height
        self.width = width
        self.segments = segments
        self.real_cols = real_cols
        self.final = final
        assert self.consistency_check(full=False)

    def consistency_check(self, full: bool = False) -> bool:
        """
        verifies that the segments are disjoint, that points haven't left the strip,
        and runs a consistency_check on each segment if <full>
        """
        if len(self.segments) == 0:
            return True
        point_sets = [segment.points() for segment in self.segments]
        assert sum(len(ps) for ps in point_sets) == len(set.union(*point_sets))
        assert all(
            max(pt[1] for pt in segment.points()) in list(range(0, self.height))
            for segment in self.segments
        )
        if full:
            assert all(
                segment.consistency_check(full=True) for segment in self.segments
            )
        return True

    def __repr__(self) -> str:
        return (
            f"State({self.height}, {self.width}, {repr(self.segments)}, "
            f"{self.real_cols}, {self.final})"
        )

    def get_next_states(
        self, probabilistic: bool, energistic: bool, full_only: bool = False
    ) -> List[Tuple["State", Weight]]:
        """
        Produces the a list of states that this state can evolve into, each with a
        weight polynomial in the variable "x" representing how many new edges are added
        in this transition.

        If <probabilistic>, then the weights of states will include probabilities.
        If <full_only>, states will be considered bad if they don't use every open slot.
        """

        # FULL_ONLY only works when width=2
        if full_only:
            assert self.width == 2, "full_only only works with width=2"

        # ENERGISTIC only works when width=4
        if energistic:
            assert self.width == 4, "energistic only work with width=4"
            assert not probabilistic

        if probabilistic:
            assert not energistic

        # We will start with a work list of <self>. Then for each segment, we pull each
        #  thing off the work list, extend that segment in all ways (both start and end
        #  if applicable), add all those possibilities to the next work list.

        assert all(
            max(pt[0] for pt in segment.points()) in set(range(0, self.width + 1))
            for segment in self.segments
        )
        if self.final:
            return []
        if len(self.segments) == 0:
            assert False
            return [(self, 1)]

        next_states: List[Tuple[State, Weight]] = []

        # a work packet consists of a state, the weight so far, an int representing the
        #  next segment in that state that should be extended, a string representing
        #  whether we should extend the start or end ("s" or "e"), and the set of open
        #  points remaining in that state (for convenience).

        WorkPacket = Tuple[State, Weight, int, str, Set[Point]]
        open_points = {(self.width, i) for i in range(self.height)}
        work_packets: List[WorkPacket] = [
            (deepcopy(self), 1, 0, "s", copy(open_points))
        ]

        while len(work_packets) > 0:
            (state, weight, seg_ind, which_seg, open_points) = work_packets.pop()

            assert which_seg in {"s", "e"}
            seg_side = "start" if which_seg == "s" else "end"  # pylint:disable=W0612

            # if seg_ind is equal to the number of segments, then we can choose to be
            #  done, or we can choose to try to add another segment
            assert seg_ind <= len(state.segments)
            if seg_ind == len(state.segments):
                assert which_seg == "s"
                next_states.append((state, weight))

            # START ADD NEW SEGMENT
            # Right here, we consider that segment [seg_ind] could be a new segment
            #  containing a single path that starts in column 2. It can go up or down,
            #  in any amount and either direction, or be a single point, as long as
            #  it does not conflict with any other segment. It cannot join an existing
            #  segment, because that would already be achieved with a later step.
            # We can't add a new segment with index 0, since segment 0 is always
            #  segment 0 by the problem definition.
            # We can't add new segments if segment 0 does have its endpoint in column 2
            #  because then it will never be able to join up.
            if (
                which_seg == "s"
                and seg_ind > 0
                and any(
                    path.points[-1][0] == self.width for path in state.segments[0].paths
                )
            ):
                assert all(
                    pt[0] == self.width for pt in open_points
                )  # just making sure
                # any point in open_points could be the start of a new path
                for start_point in open_points:
                    start_height = start_point[1]

                    # (**) New observation: a new segment is allowed to be a single
                    #  point ONLY if it is the last segment. Therefore, we start
                    #  one point ABOVE start_height when trying to go up on any
                    #  segment except the last. This seems to cut the number of
                    #  states by about 10%.
                    sh = (
                        start_height
                        if seg_ind == len(state.segments)
                        else start_height + 1
                    )

                    # first try to go up, and this is the loop in which we add the
                    #  single point path (and we need to make sure not to duplicate it
                    #  in the going down loop)

                    for new_height in range(sh, state.height):
                        if (self.width, new_height) not in open_points:
                            # once one height is bad, the rest will be as well
                            break
                        new_state = deepcopy(state)
                        new_open_points = copy(open_points)
                        new_path = Path(
                            [
                                (self.width, height)
                                for height in range(start_height, new_height + 1)
                            ]
                        )
                        for pt in new_path.points:
                            new_open_points.remove(pt)
                        new_segment = Segment([new_path])
                        new_state.segments.insert(seg_ind, new_segment)
                        assert new_state.consistency_check(full=True)
                        work_packets.append(
                            (
                                new_state,
                                weight * (x ** (new_height - start_height)),
                                seg_ind + 1,
                                "s",
                                new_open_points,
                            )
                        )
                    # now try to go down
                    for new_height in range(start_height - 1, -1, -1):
                        if (self.width, new_height) not in open_points:
                            # once one height is bad, the rest will be as well
                            break
                        new_state = deepcopy(state)
                        new_open_points = copy(open_points)
                        new_path = Path(
                            [
                                (self.width, height)
                                for height in range(start_height, new_height - 1, -1)
                            ]
                        )
                        for pt in new_path.points:
                            new_open_points.remove(pt)
                        new_segment = Segment([new_path])
                        new_state.segments.insert(seg_ind, new_segment)
                        assert new_state.consistency_check(full=True)
                        work_packets.append(
                            (
                                new_state,
                                weight * (x ** (start_height - new_height)),
                                seg_ind + 1,
                                "s",
                                new_open_points,
                            )
                        )
            # END ADD NEW SEGMENT

            # if seg_ind == len(state.segments), then we can't do anything below
            if seg_ind == len(state.segments):
                continue

            seg = state.segments[seg_ind]
            first_point = seg.paths[0].points[0]
            seg_start = first_point if first_point[0] == self.width - 1 else None
            last_point = seg.paths[-1].points[-1]
            seg_end = last_point if last_point[0] == self.width - 1 else None

            # If the start and end are the same point, then the segment consists of a
            #   single path with a single point. If this is the first segment, then it
            #   can only be the start of a path. If it's any other segment, it can only be
            #   an end (and I think there can really only be one of those in order to
            #   end up with something valid, but we don't need to worry about that
            #   here.)
            if seg_start == seg_end:
                # Note to self: it may seem like I have start/end reversed in the if-
                #   statement below, but it's correct!
                if seg_ind == 0:
                    seg_start = None
                else:
                    seg_end = None

            seg_point = seg_start if which_seg == "s" else seg_end

            if which_seg == "s":
                next_seg_ind = seg_ind
                next_seg_side = "e"
            else:
                next_seg_ind = seg_ind + 1
                next_seg_side = "s"

            if seg_point is None:
                new_state = deepcopy(state)
                new_open_points = copy(open_points)
                work_packets.append(
                    (new_state, weight, next_seg_ind, next_seg_side, new_open_points)
                )
                continue

            seg_point_extend = (self.width, seg_point[1])

            # The start point, unlike the end point, MUST, at the very least,
            #   extend horizontally, except for Segment 0 which has the start in the
            #   top-left corner and thus never moves.
            if not state.real_cols == 0 and which_seg == "s" and seg_ind == 0:
                new_state = deepcopy(state)
                new_open_points = copy(open_points)
                work_packets.append(
                    (new_state, weight, next_seg_ind, next_seg_side, new_open_points)
                )
                continue
            if which_seg == "s" and seg_point_extend not in open_points:
                continue

            # If building the end, we may choose to not move it at all
            if which_seg == "e":
                new_state = deepcopy(state)
                new_open_points = copy(open_points)
                work_packets.append(
                    (new_state, weight, next_seg_ind, next_seg_side, new_open_points)
                )

            def check_path(point: Point, path: Path, which_seg: str) -> bool:
                return point == path.points[0 if which_seg == "s" else -1]

            def add_to_path(point: Point, path: Path, which_seg: str) -> None:
                path.points.insert(0 if which_seg == "s" else len(path.points), point)
                assert path.consistency_check()

            if seg_point_extend not in open_points:
                # we can't extend because it's blocked, so we're done here
                continue

            # At the point we are definitely going to extend out, and then up or down.
            # So here we extend out, put the result on the work queue.
            state = deepcopy(state)
            open_points = copy(open_points)
            open_points.remove(seg_point_extend)
            seg = state.segments[seg_ind]
            relevant_path = seg.find_path(seg_point)
            assert check_path(seg_point, relevant_path, which_seg)

            add_to_path(seg_point_extend, relevant_path, which_seg)
            assert state.consistency_check(full=True)
            work_packets.append(
                (state, weight * x, next_seg_ind, next_seg_side, open_points)
            )

            # Now we try to extend up or down, working from <state> and <open_points>.
            cur_height = seg_point[1]
            # Now that the segment has been extended, we will try to go up any amount
            #   possible without conflicting with an existing path.
            # At each point we will also have to check if there's a compatible join
            #  that can be done, which will be a VERY annoying thing to figure out!
            # Cases where the path joins the next segment are handled later.
            for new_height in range(cur_height + 1, state.height):
                if (self.width, new_height) in open_points:
                    new_state = deepcopy(state)
                    new_seg = new_state.segments[seg_ind]
                    new_open_points = copy(open_points)
                    relevant_path = new_seg.find_path(seg_point_extend)
                    for yval in range(cur_height + 1, new_height + 1):
                        new_open_points.remove((self.width, yval))
                        add_to_path((self.width, yval), relevant_path, which_seg)
                    assert new_state.consistency_check(full=True)
                    work_packets.append(
                        (
                            new_state,
                            weight * (x ** (new_height - cur_height + 1)),
                            next_seg_ind,
                            next_seg_side,
                            new_open_points,
                        )
                    )
                else:
                    # once we hit a bad height, the rest will be bad
                    break

            # Now we will try to go DOWN any amount possible without conflicting with
            #   an existing path.
            # At each point we will also have to check if there's a compatible join
            #  that can be done, which will be a VERY annoying thing to figure out!
            # Cases where the path joins the next segment are handled later.
            for new_height in range(cur_height - 1, -1, -1):
                if (self.width, new_height) in open_points:
                    new_state = deepcopy(state)
                    new_seg = new_state.segments[seg_ind]
                    new_open_points = copy(open_points)
                    relevant_path = new_seg.find_path(seg_point_extend)
                    for yval in range(cur_height - 1, new_height - 1, -1):
                        new_open_points.remove((self.width, yval))
                        add_to_path((self.width, yval), relevant_path, which_seg)
                    assert new_state.consistency_check(full=True)
                    work_packets.append(
                        (
                            new_state,
                            weight * (x ** (cur_height - new_height + 1)),
                            next_seg_ind,
                            next_seg_side,
                            new_open_points,
                        )
                    )
                else:
                    # once we hit a bad height, the rest will be bad
                    break

            # If we are working on the end of segment i, which is not the last segment
            #  then we can connect the end of segment i to the beginning of segment i+1,
            #  as long as the start of segment i+1 is in column 1 and we have not
            #  already blocked the path. If we do this, the next work packet is allowed
            #  to once again move the end of segment i (which is the point that was
            #  previously the end of segment i+1).
            if seg_ind != len(state.segments) - 1 and which_seg == "e":
                next_first_point = state.segments[seg_ind + 1].paths[0].points[0]
                next_start = (
                    next_first_point if next_first_point[0] == self.width - 1 else None
                )

                if next_start is not None:
                    assert next_start[0] == self.width - 1
                    # the if-statement below checks that we can extend the start of
                    #  the next segment, and then that all the points properly between
                    #  are open
                    if (self.width, next_start[1]) in open_points and all(
                        (self.width, i) in open_points
                        for i in range(
                            min(cur_height, next_start[1]) + 1,
                            max(cur_height, next_start[1]),
                        )
                    ):
                        # now that we're here, we know we can link up
                        new_state = deepcopy(state)
                        new_seg = new_state.segments[seg_ind]
                        next_seg = new_state.segments[seg_ind + 1]
                        new_open_points = copy(open_points)

                        # find the right paths
                        cur_path = new_seg.find_path((self.width, cur_height))
                        next_path = next_seg.find_path(next_start)

                        # set up the loop to extend either up or down
                        if cur_height < next_start[1]:
                            loop = range(cur_height + 1, next_start[1] + 1)
                        elif cur_height > next_start[1]:
                            loop = range(cur_height - 1, next_start[1] - 1, -1)
                        else:
                            assert False
                        # extend
                        for yval in loop:
                            cur_path.points.append((self.width, yval))
                            new_open_points.remove((self.width, yval))
                        # link in next path
                        cur_path.points.extend(next_path.points)
                        assert cur_path.consistency_check()

                        next_seg.paths.remove(next_path)

                        # all other paths in next_seg join this seg
                        new_seg.paths.extend(next_seg.paths)
                        # then remove the next segment
                        new_state.segments.remove(next_seg)
                        assert new_state.consistency_check(full=True)

                        # next loop we now work from the end of this segment again
                        #   (because the end has now moved)
                        next_seg_ind -= 1
                        next_seg_side = "e"

                        # put in queue
                        work_packets.append(
                            (
                                new_state,
                                weight * (x ** (abs(cur_height - next_start[1]) + 2)),
                                next_seg_ind,
                                next_seg_side,
                                new_open_points,
                            )
                        )

        # If we're in "full_only" mode, every state gets a final copy if it has
        #  a single segment (if it's not full, it will get filtered by
        #  "is_bad_state"
        if full_only:
            new_full_final = []
            for ns, weight in next_states:
                if len(ns.segments) == 1:
                    state_copy = deepcopy(ns)
                    state_copy.final = True
                    new_full_final.append((state_copy, weight))
            next_states = next_states + new_full_final
        else:
            # check whether each state should be accepting or not
            for ns, _ in next_states:
                assert len(ns.segments) > 0
                if (
                    len(ns.segments) == 1
                    and ns.segments[0].paths[-1].points[-1][0] < self.width
                ):
                    ns.final = True

        if probabilistic:
            next_states = [(ns, ns.probability() * w) for (ns, w) in next_states]
        elif energistic:
            next_states = [(ns, ns.energy_probability() * w) for (ns, w) in next_states]
        return [
            ns.trim(w)
            for (ns, w) in next_states
            if not ns.is_bad_state(full_only=full_only)
        ]

    def probability(self) -> Fraction:
        """
        Given a state with 3 columns (i.e., before it's been trimmed), we calculate
        the probability for the steps that *originate in column 1* (in a directed
        sense). The returned fraction is the product of the probabilities for each such
        step.

        If this is a final state, is also takes into account the steps originating in
        column 2.
        """
        prob = Fraction(1)
        # We go segment-by-segment, and within each segment path-by-path, playing back
        #  the moves of the state, and at each relevant edge, compute the probability
        #  based on the number of open neighbors.
        rel_cols = {self.width - 1, self.width} if self.final else {self.width - 1}

        if self.real_cols == 0:
            open_cols = {self.width}
        elif self.real_cols == 1:
            open_cols = {self.width - 1, self.width}
        else:
            open_cols = set(range(self.width + 1))
        open_points = {(x, y) for x in open_cols for y in range(self.height)}
        if self.final:
            open_points.update({(self.width + 1, y) for y in range(self.height)})

        for segment in self.segments:
            for path in segment.paths:
                for start_point in path.points[:-1]:
                    if self.real_cols == 0 and start_point[0] == self.width - 1:
                        continue
                    open_points.remove(start_point)
                    if start_point[0] in rel_cols:
                        xpt, ypt = start_point
                        neighbors = {
                            (xpt - 1, ypt),
                            (xpt + 1, ypt),
                            (xpt, ypt - 1),
                            (xpt, ypt + 1),
                        }

                        open_neighbors = sum(1 for pt in neighbors if pt in open_points)
                        prob *= Fraction(1, open_neighbors)
                if not (self.real_cols == 0 and path.points[-1][0] == self.width - 1):
                    open_points.remove(path.points[-1])
        return prob

    def energy_probability(self) -> Weight:
        """
        Given a state with 5 columns (before being trimmed down to four), computes
        the probability of the move for all states originating in column 2 (and 3 and 4
        if this is a final state).
        """
        assert self.width == 4

        prob = sympy.sympify(1)
        # We go segment-by-segment, and within each segment path-by-path, playing back
        #  the moves of the state, and at each relevant edge, compute the probability
        #  based on the number of open neighbors and their neighbors in the path.
        rel_cols = {2, 3, 4} if self.final else {2}

        open_points = {
            (x, y)
            for x in range(self.width - self.real_cols, self.width + 1)
            for y in range(self.height)
        }
        if self.final:
            open_points.update({(self.width + 1, y) for y in range(self.height)})
        # occupied_points = set.union(*[segment.points() for segment in self.segments])
        occupied_points = set()

        def get_neighbors(pt: Point) -> List[Point]:
            return [
                (pt[0] - 1, pt[1]),
                (pt[0] + 1, pt[1]),
                (pt[0], pt[1] - 1),
                (pt[0], pt[1] + 1),
            ]

        for segment in self.segments:  # pylint: disable=R1702
            for path in segment.paths:
                for index, start_point in enumerate(path.points[:-1]):
                    if self.real_cols == 0 and start_point[0] == self.width - 1:
                        # This is the fake point that helps us initialize start states.
                        continue
                    open_points.remove(start_point)
                    occupied_points.add(start_point)
                    if start_point[0] in rel_cols:
                        neighbors = get_neighbors(start_point)

                        energy: Dict[Point, Weight] = dict()
                        for neighbor in neighbors:
                            if neighbor not in open_points:
                                # Neighbor is either occupied or out of bounds
                                continue
                            # Neighbor is available. How many neighbors does IT have in
                            #   the path?
                            nbhrs_in_path = [
                                np
                                for np in get_neighbors(neighbor)
                                if np in occupied_points
                            ]
                            energy[neighbor] = C ** len(nbhrs_in_path)

                        total_energy = sum(energy.values())

                        prob *= (energy[path.points[index + 1]] / C) / (
                            total_energy / C
                        )
                if not (self.real_cols == 0 and path.points[-1][0] == self.width - 1):
                    open_points.remove(path.points[-1])
                    occupied_points.add(path.points[-1])
        return prob.factor()

    def is_bad_state(self, full_only: bool) -> bool:
        """
        After we have produced the next_states, many of them should be thrown away
        because they are not permissible. It is assumed that this function is being
        called only on a state that has already been extended and has width 2.
        A state is bad if any of the following things are true:
         1) There is any segment except the last one that has an endpoint in any column
            except the rightmost column.
         2) The last segment has an end in column 1 but isn't maximal.

        If <full_only>, states will be considered bad if they don't use every open slot.
        """

        if full_only:
            assert self.width == 2
            points = self.points()
            if self.real_cols == 0:
                points_required = self.height + 1
            elif self.real_cols == 1:
                points_required = 2 * self.height
            else:
                points_required = 3 * self.height
            if len(points) != points_required:
                return True

        if self.final and len(self.segments) > 1:
            assert False, "We should never get here."
            return True

        ## TESTING: A state is bad if it has a non-last segment that is a
        ##   single path, with a single point, in the rightmost col
        for segment in self.segments[:-1]:
            segpts = segment.points()
            if len(segpts) == 1 and next(iter(segpts))[0] == 2:
                return True

        # a state is bad if there is any segment except the last one that has an
        #   endpoint in any column except the rightmost column
        for segment in self.segments[:-1]:
            rightmost_endpoint = max(segment.paths, key=lambda p: p.points[-1][0])
            col = rightmost_endpoint.points[-1][0]
            if col != self.width:
                return True

        # last segment: any end in column 1 must be maximal
        if self.final and len(self.segments) == 1:
            end_points = [
                path.points[-1]
                for path in self.segments[0].paths
                if path.points[-1][0] in {self.width - 1, self.width}
            ]
        else:
            end_points = [
                path.points[-1]
                for path in self.segments[-1].paths
                if path.points[-1][0] == self.width - 1
            ]
        assert len(end_points) <= 1
        if len(end_points) == 1:
            end_point = end_points[0]
            nearby_points = {(end_point[0] - 1, end_point[1])}
            if end_point[0] == self.width - 1:
                nearby_points.add((end_point[0] + 1, end_point[1]))
            if end_point[1] > 0:
                nearby_points.add((end_point[0], end_point[1] - 1))
            if end_point[1] < self.height - 1:
                nearby_points.add((end_point[0], end_point[1] + 1))
            if self.real_cols in {0, 1}:
                if self.real_cols == 0 or end_point[0] == self.width - 1:
                    nearby_points.remove((end_point[0] - 1, end_point[1]))
            if any(
                all(np not in segment.points() for segment in self.segments)
                for np in nearby_points
            ):
                return True
        return False

    def trim(self, weight: Weight) -> Tuple["State", Weight]:
        """
        Shifts all segments left by 1 unit. This may cause a path to break into several
        paths (which all remain in the same segment). Increases self.real_cols by 1
        up until it equals self.width.
        """
        next_real_cols = min(self.width, self.real_cols + 1)

        trimmed_segments = [segment.trim() for segment in self.segments]

        if self.real_cols == 0:
            trimmed_segments[0].paths[0].points.pop(0)
            weight = weight / x

        if any(len(segment.paths) == 0 for segment in trimmed_segments):
            assert len(self.segments) == 1, repr(self)
            return (
                State(self.height, self.width, [], next_real_cols, self.final),
                weight,
            )
        new_state = State(
            self.height, self.width, trimmed_segments, next_real_cols, self.final
        )
        assert new_state.consistency_check(full=True)
        return new_state, weight

    def points(self) -> Set[Point]:
        """
        returns the set of points contained in the segments of this state
        """
        if len(self.segments) == 0:
            return set()
        return set.union(*[set(segment.points()) for segment in self.segments])

    def flip(self) -> "State":
        """
        returns an upside-down version of the state
        """

        def flip_point(pt: Point) -> Point:
            return (pt[0], self.height - 1 - pt[1])

        return State(
            self.height,
            self.width,
            [
                Segment(
                    [
                        Path([flip_point(pt) for pt in path.points])
                        for path in segment.paths
                    ]
                )
                for segment in self.segments
            ],
            self.real_cols,
            self.final,
        )

    @staticmethod
    def init_state(height: int, width: int) -> "State":
        """returns the initial state that will kick off the finite state machine"""
        return State(
            height, width, [Segment([Path([(width - 1, height - 1)])])], 0, False
        )

    def __str__(self) -> str:
        end_chr = "|" if self.final else ":"
        if len(self.segments) == 0:
            return "---\n" + (f": {end_chr}\n" * (3 * (self.height - 1) + 1)) + "---"
        colors = [2, 1, 4, 3, 5, 6, 8, 47, 50, 52, 55, 58, 68, 95, 122, 195, 229, 237]
        assert len(self.segments) <= len(
            colors
        ), f"We only have enough colors for {len(colors)} paths!"
        start_color = "\u001b[48;5;#m"
        end_color = "\u001b[0m"
        st_col = {
            i: start_color.replace("#", str(colors[i]))
            for i in range(len(self.segments))
        }

        # (x,y) -> i means the point (x,y) is in segment i
        point_assignments: Dict[Point, int] = dict()
        for index, segment in enumerate(self.segments):
            for path in segment.paths:
                for point in path.points:
                    point_assignments[point] = index

        # (x,y) -> (i,"<") means the path from (x+1,y) to (x,y) is in segment i
        # (x,y) -> (i,">") means the path from (x,y) to (x+1,y) is in segment i
        horizontal_path_assignments: Dict[Point, Tuple[int, str]] = dict()

        # (x,y) -> (i,"^") means the path from (x,y) to (x,y+1) is in segment i
        # (x,y) -> (i,"v") means the path from (x,y+1) to (x,y) is in segment i
        vertical_path_assignments: Dict[Point, Tuple[int, str]] = dict()

        for index, segment in enumerate(self.segments):
            for path in segment.paths:
                for pt1, pt2 in zip(path.points[:-1], path.points[1:]):
                    if pt1[1] == pt2[1]:
                        if pt1[0] == pt2[0] - 1:
                            horizontal_path_assignments[pt1] = (index, ">")
                        elif pt1[0] == pt2[0] + 1:
                            horizontal_path_assignments[pt2] = (index, "<")
                        else:
                            assert False, repr(path)
                    elif pt1[0] == pt2[0]:
                        if pt1[1] == pt2[1] - 1:
                            vertical_path_assignments[pt1] = (index, "^")
                        elif pt1[1] == pt2[1] + 1:
                            vertical_path_assignments[pt2] = (index, "v")
                        else:
                            assert False, repr(path)
                    else:
                        assert False, repr(path)

        # Now we have all info set up!
        # How wide and tall does the figure have to be?
        max_x = max(1, max(pt[0] for pt in point_assignments.keys()))
        # max_x = self.width - 1
        # print(list(point_assignments.keys()))

        S = "-" * (5 + 5 * max_x) + "\n"
        for y_val in range(self.height - 1, -1, -1):
            # print(y_val)
            S += ": "

            # first do the horizontal row of points
            for x_val in range(max_x):
                if (x_val, y_val) in point_assignments:
                    index = point_assignments[(x_val, y_val)]
                    S += f"{st_col[index]}*{end_color}"
                else:
                    S += "*"
                if (x_val, y_val) in horizontal_path_assignments:
                    (index, arrow) = horizontal_path_assignments[(x_val, y_val)]
                    S += f"{st_col[index]} {arrow*2} {end_color}"
                else:
                    S += "    "
            if (max_x, y_val) in point_assignments:
                index = point_assignments[(max_x, y_val)]
                S += f"{st_col[index]}*{end_color}"
            else:
                S += "*"
            S += f" {end_chr}"

            # now we do the arrows below it, but not when y_val = 0
            if y_val > 0:
                S += "\n: "
                ver_arrows = []
                for x_val in range(max_x + 1):
                    if (x_val, y_val - 1) in vertical_path_assignments:
                        (index, arrow) = vertical_path_assignments[(x_val, y_val - 1)]
                        ver_arrows.append(f"{st_col[index]}{arrow}{end_color}")
                    else:
                        ver_arrows.append(" ")
                S += "    ".join(ver_arrows)
                S += f" {end_chr}\n: "
                S += "    ".join(ver_arrows)
                S += f" {end_chr}\n"

        S += "\n" + "-" * (5 + 5 * max_x)

        return S

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, State):
            return NotImplemented
        return (
            self.height == other.height
            and self.width == other.width
            and self.segments == other.segments
            and self.real_cols == other.real_cols
            and self.final == other.final
        )

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, State):
            return NotImplemented
        return hash(self) < hash(other)

    def __hash__(self) -> int:
        return hash(
            (self.height, self.width, tuple(self.segments), self.real_cols, self.final)
        )
