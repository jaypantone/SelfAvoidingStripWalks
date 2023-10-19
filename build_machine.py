from typing import Dict, List, Set, Tuple, Union

from fractions import Fraction
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
        # return state
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

    CFSM = CombinatorialFSM(y)
    CFSM.set_start(init_state)
    CFSM.set_accepting([state for state in done if state.final])

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
        displacement_weight = 1 if old_state == init_state else y
        for new_state, weight in next_states:
            CFSM.add_transition(
                # old_state, new_state, sympy.sympify(weight).subs(x, y) * x
                old_state,
                new_state,
                sympy.sympify(weight * displacement_weight),
            )

    # minimize until no change
    num_states = len(CFSM.states)
    print(
        f"Machine has {num_states} states and "
        f"{len(CFSM.transition_weights)} transitions."
    )
    # return CFSM
    # minimized = CFSM.minimize()
    # while len(minimized.states) != num_states:
    #     num_states = len(minimized.states)
    #     print(
    #         f"Machine has {num_states} states and "
    #         f"{len(minimized.transition_weights)} transitions."
    #     )
    #     minimized = minimized.minimize()

    print("Minimizing with new Moore method.")
    minimized = CFSM.moore_minimize(verbose=True)
    print(
        f"Machine has {len(minimized.states)} states and "
        f"{len(minimized.transition_weights)} transitions."
    )

    with open(
        f"results/walks_{height}_{width}{'_prob' if probabilistic else ''}"
        f"{'_energ' if energistic else ''}{'_full_only' if full_only else ''}.txt",
        "w",
    ) as f:
        minimized.write_to_maple_file(f)

    return minimized


def run_non_prob_tests() -> None:
    """testing routine"""

    print("[height=2, nonprob] checking 2,2: ")
    M2 = build_machine(2, 2, False, False, False)
    assert (
        M2.enumeration(len(test_data.M2nonprob), quiet=True)[1:] == test_data.M2nonprob
    )
    print("good\n")

    print("[height=3, nonprob] checking 3,2: ")
    M3 = build_machine(3, 2, False, False, False)
    assert (
        M3.enumeration(len(test_data.M3nonprob), quiet=True)[1:] == test_data.M3nonprob
    )
    print("good\n")

    print("[height=4, nonprob] checking 4,2: ")
    M4 = build_machine(4, 2, False, False, False)
    assert (
        M4.enumeration(len(test_data.M4nonprob), quiet=True)[1:] == test_data.M4nonprob
    )
    print("good\n")


def run_prob_tests() -> None:
    """testing routine"""

    print("[height=2, prob] checking 2,2: ")
    M2 = build_machine(2, 2, True, False, False)
    assert M2.enumeration(len(test_data.M2prob), quiet=True)[1:] == test_data.M2prob
    print("good\n")

    print("[height=3, prob] checking 3,2: ")
    M3 = build_machine(3, 2, True, False, False)
    assert M3.enumeration(len(test_data.M3prob), quiet=True)[1:] == test_data.M3prob
    print("good\n")

    print("[height=4, prob] checking 4,2: ")
    M4 = build_machine(4, 2, True, False, False)
    assert M4.enumeration(len(test_data.M4prob), quiet=True)[1:] == test_data.M4prob
    print("good\n")


def run_full_tests() -> None:
    """testing routine"""

    print("[height=2, full] checking 2,2: ")
    M2 = build_machine(2, 2, False, False, True)
    assert M2.enumeration(len(test_data.M2full), quiet=True)[1:] == test_data.M2full
    print("good\n")

    print("[height=3, full] checking 3,2: ")
    M3 = build_machine(3, 2, False, False, True)
    assert M3.enumeration(len(test_data.M3full), quiet=True)[1:] == test_data.M3full
    print("good\n")

    print("[height=4, full] checking 4,2: ")
    M4 = build_machine(4, 2, False, False, True)
    assert M4.enumeration(len(test_data.M4full), quiet=True)[1:] == test_data.M4full
    print("good\n")

    print("[height=5, full] checking 5,2: ")
    M5 = build_machine(5, 2, False, False, True)
    assert M5.enumeration(len(test_data.M5full), quiet=True)[1:] == test_data.M5full
    print("good\n")


def run_energistic_tests() -> None:
    """testing routine"""

    # We need to substitute something for C because otherwise
    # sympy is incapable of factoring
    Csub = Fraction(17, 31)

    print("[height=2, energistic] checking 2,4: ")
    M2 = build_machine(2, 4, False, True, False)
    data1 = M2.enumeration(len(test_data.M2energistic), quiet=True)[1:]
    data2 = test_data.M2energistic
    assert all(
        (sympy.sympify(d1).subs(C, Csub) - sympy.sympify(d2).subs(C, Csub)).factor()
        == 0
        for (d1, d2) in zip(data1, data2)
    )
    print("good\n")

    print("[height=3, energistic] checking 3,4: ")
    M3 = build_machine(3, 4, False, True, False)
    data1 = M3.enumeration(len(test_data.M3energistic), quiet=True)[1:]
    data2 = test_data.M3energistic
    assert all(
        (sympy.sympify(d1).subs(C, Csub) - sympy.sympify(d2).subs(C, Csub)).factor()
        == 0
        for (d1, d2) in zip(data1, data2)
    )
    print("good\n")


def run_all_tests() -> None:
    """run all tests"""
    run_non_prob_tests()
    run_prob_tests()
    run_full_tests()
    run_energistic_tests()
