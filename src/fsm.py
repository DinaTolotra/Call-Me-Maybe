class State:
    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def __eq__(self, value: object) -> bool:
        if isinstance(value, State):
            return self.name == value.name
        return False

    def __hash__(self) -> int:
        return hash(self.name)


class TransitionTable:
    def __init__(self) -> None:
        self._table: dict[
            tuple[State, State],
            set[int]
        ] = {}

    def get_allowed_token(
            self,
            state: State,
            ) -> set[int]:
        result = set[int]()
        for src, dst in self._table:
            val = self._table[(src, dst)]
            if src == state:
                result = result.union(val)
        return result

    def get_dst(self, state: State, token: int) -> State | None:
        for src, dst in self._table:
            if src == state and token in self._table[(src, dst)]:
                return dst
        return None

    def add_transition(
            self,
            src: State,
            dst: State,
            token: set[int]
            ) -> None:
        temp = self._table.get((src, dst), set())
        self._table[(src, dst)] = temp.union(token)


class Grammar:
    def __init__(
            self,
            transition_table: TransitionTable,
            initial_state: State,
            final_state: State,
            max_tokens: int = 128
            ) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be greater than zero")
        self._table = transition_table
        self._state = initial_state
        self._final_state = final_state
        self._result = list[int]()
        self._max_tokens = max_tokens
        self._token_count = 0

    @property
    def result(self) -> list[int]:
        return self._result

    @property
    def state(self) -> State:
        return self._state

    @property
    def token(self) -> set[int]:
        return self._table.get_allowed_token(self.state)

    @property
    def has_ended(self) -> bool:
        return self.state == self._final_state

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    @property
    def limit_reached(self) -> bool:
        return self._token_count >= self._max_tokens

    def consume(self, token: int) -> bool:
        if self.limit_reached:
            return False
        self._token_count += 1
        if token in self._table.get_allowed_token(self._state):
            state = self._table.get_dst(self.state, token)
            self._state = state if state else self._final_state
            self._result.append(token)
            return True
        return False
