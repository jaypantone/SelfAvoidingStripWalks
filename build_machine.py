from typing import Dict, List, Set, Tuple, Union

import sympy  # type: ignore
from finite_state_machines import CombinatorialFSM

import test_data
import walk
from state import State

x, y, C = sympy.symbols("x y C")
Weight = Union[sympy.polys.polytools.Poly, sympy.Expr]


class SymTracker:
    """
    Tracks which state of {state, state.flip()} is minimal and returns that so it can
    always be used as the representative.
    """

    def __init__(self) -> None:
        self._map: Dict[State, State] = dict()
        self.fake_acceptance = State(0, 0, [], 0, final=True)

    def get(self, state: State) -> State:
        """
        Returns state or state.flip(), whichever is <. Caches the result.
        """
        if state.final:
            # fake acceptance state to stand in for all acceptance states
            return self.fake_acceptance
        if state not in self._map:
            flipped_state = state.flip()
            min_state = min(state, flipped_state)
            self._map[state] = min_state
            self._map[flipped_state] = min_state
        return self._map[state]


def build_machine(
    height: int, width: int, probabilistic: bool, energistic: bool, full_only: bool
) -> CombinatorialFSM:
    """
    Builds the CombinatorialFSM that counts various kinds of walks.
    At most one of <probabilistic>, <energistic>, and <full_only> can be true.
    Returns a CombinatorialFSM object and also writes a Maple file to disk to compute
       the generating function.
    """
    sym = SymTracker()
    init_state = sym.get(State.init_state(height, width))
    todo: Set[State] = {init_state}
    done: Set[State] = set()
    state_to_next_states: Dict[
        State,
        List[Tuple[State, Weight]],
    ] = dict()
    first = True
    back_line = "\x1b[A"

    while len(todo) > 0:
        print(
            f"{back_line if not first else ''}Todo: {len(todo)} -- Done: {len(done)}"
            + " " * 20
        )
        first = False
        n1 = todo.pop()
        done.add(n1)
        ns = n1.get_next_states(probabilistic, energistic, full_only)
        ns = [(sym.get(s), w) for (s, w) in ns]
        state_to_next_states[n1] = ns
        for n2 in ns:
            if n2[0] not in done:
                if n2[0].final:
                    done.add(n2[0])
                else:
                    todo.add(n2[0])
    print(
        f"{back_line if not first else ''}Todo: {len(todo)} -- Done: {len(done)}"
        + " " * 20
    )

    CFSM = CombinatorialFSM(x)
    CFSM.set_start(init_state)
    CFSM.set_accepting([init_state] + [state for state in done if state.final])

    # accepting = [start]
    # for state in state_list:
    #     if state.final:
    #         if any(
    #             pt[0] == state.width - 1
    #             for segment in state.segments
    #             for path in segment.paths
    #             for pt in path.points
    #         ):
    #             accepting.append(state_to_index[state])

    for old_state, next_states in state_to_next_states.items():
        for new_state, weight in next_states:
            CFSM.add_transition(
                old_state, new_state, sympy.sympify(weight).subs(x, y) * x
            )

    # minimize until no change
    num_states = len(CFSM.states)
    print(
        f"Machine has {num_states} states and "
        f"{len(CFSM.transition_weights)} transitions. Minimizing..."
    )
    minimized = CFSM.minimize()
    while len(minimized.states) != num_states:
        num_states = len(minimized.states)
        print(
            f"Machine has {num_states} states and "
            f"{len(minimized.transition_weights)} transitions. Minimizing..."
        )
        minimized = minimized.minimize()

    with open(
        f"walks_{height}_{width}{'_prob' if probabilistic else ''}"
        f"{'_energ' if energistic else ''}{'_full_only' if full_only else ''}.txt",
        "w",
    ) as f:
        minimized.write_to_maple_file(f)

    return minimized


def run_non_prob_tests() -> None:
    """testing routine"""

    print(
        "[height=2, nonprob] checking stored data vs brute generation: ",
        end="",
        flush=True,
    )
    assert test_data.M2nonprob[1:] == [
        sum(
            c * y ** i
            for (i, c) in enumerate(walk.count_maximal_anchored_walks_by_length(2, i))
        )
        for i in range(1, 13)
    ]
    print("good.")
    print("[height=2, nonprob] checking 2,2: ", end="", flush=True)
    M22 = build_machine(2, 2, False, False, False)
    assert M22.enumeration(12, quiet=True) == test_data.M2nonprob
    print(
        f"good. {len(M22.states)} states with "
        f"{len(M22.transition_weights)} transitions."
    )
    print("[height=2, nonprob] checking 2,3: ", end="", flush=True)
    M23 = build_machine(2, 3, False, False, False)
    assert M23.enumeration(12, quiet=True) == test_data.M2nonprob
    print(
        f"good. {len(M23.states)} states with "
        f"{len(M23.transition_weights)} transitions."
    )
    print("[height=2, nonprob] checking 2,4: ", end="", flush=True)
    M24 = build_machine(2, 4, False, False, False)
    assert M24.enumeration(12, quiet=True) == test_data.M2nonprob
    print(
        f"good. {len(M24.states)} states with "
        f"{len(M24.transition_weights)} transitions."
    )
    print("[height=2, nonprob] checking 2,5: ", end="", flush=True)
    M25 = build_machine(2, 5, False, False, False)
    assert M25.enumeration(12, quiet=True) == test_data.M2nonprob
    print(
        f"good. {len(M25.states)} states with "
        f"{len(M25.transition_weights)} transitions."
    )

    print(
        "[height=3, nonprob] checking stored data vs brute generation: ",
        end="",
        flush=True,
    )
    assert test_data.M3nonprob[1:] == [
        sum(
            c * y ** i
            for (i, c) in enumerate(walk.count_maximal_anchored_walks_by_length(3, i))
        )
        for i in range(1, 13)
    ]
    print("good.")
    print("[height=3, nonprob] checking 3,2: ", end="", flush=True)
    M32 = build_machine(3, 2, False, False, False)
    assert M32.enumeration(12, quiet=True) == test_data.M3nonprob
    print(
        f"good. {len(M32.states)} states with "
        f"{len(M32.transition_weights)} transitions."
    )
    print("[height=3, nonprob] checking 3,3: ", end="", flush=True)
    M33 = build_machine(3, 3, False, False, False)
    assert M33.enumeration(12, quiet=True) == test_data.M3nonprob
    print(
        f"good. {len(M33.states)} states with "
        f"{len(M33.transition_weights)} transitions."
    )
    print("[height=3, nonprob] checking 3,4: ", end="", flush=True)
    M34 = build_machine(3, 4, False, False, False)
    assert M34.enumeration(12, quiet=True) == test_data.M3nonprob
    print(
        f"good. {len(M34.states)} states with "
        f"{len(M34.transition_weights)} transitions."
    )
    print("[height=3, nonprob] checking 3,5: ", end="", flush=True)
    M35 = build_machine(3, 5, False, False, False)
    assert M35.enumeration(12, quiet=True) == test_data.M3nonprob
    print(
        f"good. {len(M35.states)} states with "
        f"{len(M35.transition_weights)} transitions."
    )

    print(
        "[height=4, nonprob] checking stored data vs brute generation: ",
        end="",
        flush=True,
    )
    assert test_data.M4nonprob[1:10] == [
        sum(
            c * y ** i
            for (i, c) in enumerate(walk.count_maximal_anchored_walks_by_length(4, i))
        )
        for i in range(1, 10)
    ]
    print("good.")
    print("[height=4, nonprob] checking 4,2: ", end="", flush=True)
    M42 = build_machine(4, 2, False, False, False)
    assert M42.enumeration(10, quiet=True) == test_data.M4nonprob
    print(
        f"good. {len(M42.states)} states with "
        f"{len(M42.transition_weights)} transitions."
    )
    print("[height=4, nonprob] checking 4,3: ", end="", flush=True)
    M43 = build_machine(4, 3, False, False, False)
    assert M43.enumeration(10, quiet=True) == test_data.M4nonprob
    print(
        f"good. {len(M43.states)} states with "
        f"{len(M43.transition_weights)} transitions."
    )


def run_prob_tests() -> None:
    """testing routine"""

    print(
        "[height=2, prob] checking stored data vs brute generation: ",
        end="",
        flush=True,
    )
    assert test_data.M2prob[1:] == [
        sum(
            c * y ** i
            for (i, c) in enumerate(
                walk.count_probabilistic_maximal_anchored_walks_by_length(2, i)
            )
        )
        for i in range(1, 13)
    ]
    print("good.")
    print("[height=2, prob] checking 2,2: ", end="", flush=True)
    M22 = build_machine(2, 2, True, False, False)
    assert M22.enumeration(12, quiet=True) == test_data.M2prob
    print(
        f"good. {len(M22.states)} states with "
        f"{len(M22.transition_weights)} transitions."
    )
    print("[height=2, prob] checking 2,3: ", end="", flush=True)
    M23 = build_machine(2, 3, True, False, False)
    assert M23.enumeration(12, quiet=True) == test_data.M2prob
    print(
        f"good. {len(M23.states)} states with "
        f"{len(M23.transition_weights)} transitions."
    )
    print("[height=2, prob] checking 2,4: ", end="", flush=True)
    M24 = build_machine(2, 4, True, False, False)
    assert M24.enumeration(12, quiet=True) == test_data.M2prob
    print(
        f"good. {len(M24.states)} states with "
        f"{len(M24.transition_weights)} transitions."
    )
    print("[height=2, prob] checking 2,5: ", end="", flush=True)
    M25 = build_machine(2, 5, True, False, False)
    assert M25.enumeration(12, quiet=True) == test_data.M2prob
    print(
        f"good. {len(M25.states)} states with "
        f"{len(M25.transition_weights)} transitions."
    )

    print(
        "[height=3, prob] checking stored data vs brute generation: ",
        end="",
        flush=True,
    )
    assert test_data.M3prob[1:] == [
        sum(
            c * y ** i
            for (i, c) in enumerate(
                walk.count_probabilistic_maximal_anchored_walks_by_length(3, i)
            )
        )
        for i in range(1, 13)
    ]
    print("good.")
    print("[height=3, prob] checking 3,2: ", end="", flush=True)
    M32 = build_machine(3, 2, True, False, False)
    assert M32.enumeration(12, quiet=True) == test_data.M3prob
    print(
        f"good. {len(M32.states)} states with "
        f"{len(M32.transition_weights)} transitions."
    )
    print("[height=3, prob] checking 3,3: ", end="", flush=True)
    M33 = build_machine(3, 3, True, False, False)
    assert M33.enumeration(12, quiet=True) == test_data.M3prob
    print(
        f"good. {len(M33.states)} states with "
        f"{len(M33.transition_weights)} transitions."
    )
    print("[height=3, prob] checking 3,4: ", end="", flush=True)
    M34 = build_machine(3, 4, True, False, False)
    assert M34.enumeration(12, quiet=True) == test_data.M3prob
    print(
        f"good. {len(M34.states)} states with "
        f"{len(M34.transition_weights)} transitions."
    )
    print("[height=3, prob] checking 3,5: ", end="", flush=True)
    M35 = build_machine(3, 5, True, False, False)
    assert M35.enumeration(12, quiet=True) == test_data.M3prob
    print(
        f"good. {len(M35.states)} states with "
        f"{len(M35.transition_weights)} transitions."
    )

    print(
        "[height=4, prob] checking stored data vs brute generation: ",
        end="",
        flush=True,
    )
    assert test_data.M4prob[1:10] == [
        sum(
            c * y ** i
            for (i, c) in enumerate(
                walk.count_probabilistic_maximal_anchored_walks_by_length(4, i)
            )
        )
        for i in range(1, 10)
    ]
    print("good.")
    print("[height=4, prob] checking 4,2: ", end="", flush=True)
    M42 = build_machine(4, 2, True, False, False)
    assert M42.enumeration(10, quiet=True) == test_data.M4prob
    print(
        f"good. {len(M42.states)} states with "
        f"{len(M42.transition_weights)} transitions."
    )
    print("[height=4, prob] checking 4,3: ", end="", flush=True)
    M43 = build_machine(4, 3, True, False, False)
    assert M43.enumeration(8, quiet=True) == test_data.M4prob[:9]
    print(
        f"good. {len(M43.states)} states with "
        f"{len(M43.transition_weights)} transitions."
    )


def run_full_tests() -> None:
    """testing routine"""

    print(
        "[height=2, full] checking stored data vs brute generation: ",
        end="",
        flush=True,
    )
    assert test_data.M2full[1:] == [
        walk.count_maximal_anchored_full_walks(2, i) * y ** (2 * i - 1)
        for i in range(1, 13)
    ]
    print("good.")
    print("[height=2, full] checking 2,2: ", end="", flush=True)
    M22 = build_machine(2, 2, False, False, True)
    assert M22.enumeration(12, quiet=True) == test_data.M2full
    print(
        f"good. {len(M22.states)} states with "
        f"{len(M22.transition_weights)} transitions."
    )

    print(
        "[height=3, full] checking stored data vs brute generation: ",
        end="",
        flush=True,
    )
    assert test_data.M3full[1:] == [
        walk.count_maximal_anchored_full_walks(3, i) * y ** (3 * i - 1)
        for i in range(1, 13)
    ]
    print("good.")
    print("[height=3, full] checking 3,2: ", end="", flush=True)
    M32 = build_machine(3, 2, False, False, True)
    assert M32.enumeration(12, quiet=True) == test_data.M3full
    print(
        f"good. {len(M32.states)} states with "
        f"{len(M32.transition_weights)} transitions."
    )

    print(
        "[height=4, full] checking stored data vs brute generation: ",
        end="",
        flush=True,
    )
    assert test_data.M4full[1:10] == [
        walk.count_maximal_anchored_full_walks(4, i) * y ** (4 * i - 1)
        for i in range(1, 10)
    ]
    print("good.")
    print("[height=4, full] checking 4,2: ", end="", flush=True)
    M42 = build_machine(4, 2, False, False, True)
    assert M42.enumeration(10, quiet=True) == test_data.M4full
    print(
        f"good. {len(M42.states)} states with "
        f"{len(M42.transition_weights)} transitions."
    )

    print(
        "[height=5, full] checking stored data vs brute generation: ",
        end="",
        flush=True,
    )
    assert test_data.M5full[1:8] == [
        walk.count_maximal_anchored_full_walks(5, i) * y ** (5 * i - 1)
        for i in range(1, 8)
    ]
    print("good.")
    print("[height=5, full] checking 5,2: ", end="", flush=True)
    M52 = build_machine(5, 2, False, False, True)
    assert M52.enumeration(12, quiet=True) == test_data.M5full
    print(
        f"good. {len(M52.states)} states with "
        f"{len(M52.transition_weights)} transitions."
    )


def run_energistic_tests() -> None:
    """testing routine"""

    print(
        "[height=2, energistic] checking stored data vs brute generation: ",
        end="",
        flush=True,
    )
    assert all(
        (
            test_data.M2energistic[i]
            - sum(
                term * y ** j
                for (j, term) in enumerate(
                    walk.count_energistic_maximal_anchored_walks_by_length(2, i)
                )
            )
        ).subs(C, 7)
        == 0
        for i in range(1, len(test_data.M2energistic))
    )
    print("good.")
    print("[height=2, energistic] checking 2,4: ", end="", flush=True)
    M24 = build_machine(2, 4, False, True, False)
    enum24 = M24.enumeration(len(test_data.M2energistic) - 1, quiet=True)
    assert all(
        sympy.sympify(enum24[i] - test_data.M2energistic[i]).subs(C, 7) == 0
        for i in range(len(test_data.M2energistic))
    )
    print(
        f"good. {len(M24.states)} states with "
        f"{len(M24.transition_weights)} transitions."
    )

    print(
        "[height=3, energistic] checking stored data vs brute generation: ",
        end="",
        flush=True,
    )
    assert all(
        (
            test_data.M3energistic[i]
            - sum(
                term * y ** j
                for (j, term) in enumerate(
                    walk.count_energistic_maximal_anchored_walks_by_length(3, i)
                )
            )
        ).subs(C, 7)
        == 0
        for i in range(1, len(test_data.M3energistic))
    )
    print("good.")
    print("[height=3, energistic] checking 3,4: ", end="", flush=True)
    M34 = build_machine(3, 4, False, True, False)
    enum34 = M34.enumeration(7, quiet=True)
    assert all(
        sympy.sympify(enum34[i] - test_data.M3energistic[i]).subs(C, 7) == 0
        for i in range(8)
    )
    print(
        f"good. {len(M34.states)} states with "
        f"{len(M34.transition_weights)} transitions."
    )

    print(
        "[height=4, energistic] checking stored data vs brute generation: ",
        end="",
        flush=True,
    )
    assert all(
        (
            test_data.M4energistic[i]
            - sum(
                term * y ** j
                for (j, term) in enumerate(
                    walk.count_energistic_maximal_anchored_walks_by_length(4, i)
                )
            )
        ).subs(C, 7)
        == 0
        for i in range(1, len(test_data.M4energistic))
    )
    print("good.")
    print("[height=4, energistic] checking 4,4: ", end="", flush=True)
    M44 = build_machine(4, 4, False, True, False)
    enum44 = M44.enumeration(len(test_data.M4energistic) - 1, quiet=True)
    assert all(
        sympy.sympify(enum44[i] - test_data.M4energistic[i]).subs(C, 7) == 0
        for i in range(len(test_data.M4energistic))
    )
    print(
        f"good. {len(M44.states)} states with "
        f"{len(M44.transition_weights)} transitions."
    )


def run_all_tests() -> None:
    """run all tests"""
    run_non_prob_tests()
    run_prob_tests()
    run_full_tests()
    run_energistic_tests()
