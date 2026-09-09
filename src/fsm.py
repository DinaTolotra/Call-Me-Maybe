from enum import Enum, auto


t_token_set = set[int]


class State(Enum):
    GENERATING = auto()
    STOPING = auto()


class TransitionTable:
    def __init__(self) -> None:
        self._table: dict[
            State,
            list[tuple[t_token_set, State]]
        ] = {}

    def get_transition_list(
            self,
            state: State
            ) -> list[tuple[t_token_set, State]]:
        return self._table[state]

    def add_transition(
            self,
            src: State,
            dst: State | None,
            token: t_token_set | None
            ) -> None:
        l = self._table.get(src, [])
        if dst is not None and token is not None:
            l.append((token, dst))
        self._table[src] = l


class Grammar:
    def __init__(
            self,
            transition_table: TransitionTable,
            initial_state: State
            ) -> None:
        self._table = transition_table
        self._state = initial_state
        self._result = list[int]()

    @property
    def result(self) -> list[int]:
        return self._result

    @property
    def state(self) -> State:
        return self._state

    @property
    def possible_token(self) -> set[int]:
        token_set = t_token_set()
        for token, _ in self._table.get_transition_list(self.state):
            token_set = token_set.union(token)
        return token_set

    def consume(self, token: int) -> bool:
        for token_set, state in self._table.get_transition_list(self._state):
            if token in token_set:
                self._result.append(token)
                self._state = state
                return True
        return False
