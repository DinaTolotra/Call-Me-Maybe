from string import digits, printable

from src.llm_model import Model
from src.model import FnDef
import src.fsm as fsm


state = {
    "stoping": fsm.State("stoping"),
    "generating": fsm.State("generating"),
    "sign": fsm.State("sign"),
    "digit": fsm.State("digit"),
    "opening_quote": fsm.State("opening_quote"),
    "closing_quote": fsm.State("closing_quote"),
    "non-quote": fsm.State("non-quote")
}


class TransitionTableFactory:
    @classmethod
    def create_fn_name_table(cls, model: Model, fn_list: list[FnDef]) -> fsm.TransitionTable:
        table = fsm.TransitionTable()
        fn_name_token = list[int]()
        fn_list.append(FnDef(
            name = "undefined",
            description = "None of the functions listed",
            parameters = {}, returns = {}
        ))
        for fn in fn_list:
            token = model.encode(fn.name)
            fn_name_token.extend(token)
        table.add_transition(
            state["generating"],
            state["generating"],
            set(fn_name_token)
        )
        table.add_transition(
            state["generating"],
            state["stoping"],
            set(model.encode(";"))
        )
        return table

    @classmethod
    def create_number_table(cls, model: Model) -> fsm.TransitionTable:
        table = fsm.TransitionTable()
        digit_token = list[int]()
        for d in digits:
            token = model.encode(d)
            digit_token.extend(token)
        table.add_transition(
            state["digit"],
            state["digit"],
            set(digit_token)
        )
        table.add_transition(
            state["digit"],
            state["stoping"],
            set(model.encode(";"))
        )
        return table

    @classmethod
    def create_string_table(cls, model: Model) -> fsm.TransitionTable:
        table = fsm.TransitionTable()
        non_quote_token = list[int]()
        for token in range(model.vocab_size):
            text = model.decode([token])
            if "\n" not in text and "\"" not in text and text:
                non_quote_token.append(token)
        table.add_transition(
            state["opening_quote"],
            state["non-quote"],
            set(model.encode("\""))
        )
        table.add_transition(
            state["non-quote"],
            state["non-quote"],
            set(non_quote_token)
        )
        table.add_transition(
            state["non-quote"],
            state["closing_quote"],
            set(model.encode("\";"))
        )
        return table


class GrammarFactory:
    @classmethod
    def create_fn_name_grammar(cls, model: Model, fn_list: list[FnDef]) -> fsm.Grammar:
        return fsm.Grammar(
            TransitionTableFactory.create_fn_name_table(model, fn_list),
            state["generating"], state["stoping"]
        )

    @classmethod
    def create_number_grammar(cls, model: Model) -> fsm.Grammar:
        return fsm.Grammar(
            TransitionTableFactory.create_number_table(model),
            state["digit"], state["stoping"]
        )

    @classmethod
    def create_string_grammar(cls, model: Model) -> fsm.Grammar:
        return fsm.Grammar(
            TransitionTableFactory.create_string_table(model),
            state["opening_quote"], state["closing_quote"]
        )
