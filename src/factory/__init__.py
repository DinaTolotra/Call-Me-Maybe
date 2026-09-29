from src.llm_model import Model
from src.model import FnDef
import src.fsm as fsm


class TransitionTableFactory:
    state = {
        "generating": fsm.State("generating"),
        "stoping": fsm.State("stoping")
    }

    @classmethod
    def create(cls, model: Model, fn_list: list[FnDef]) -> fsm.TransitionTable:
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
            cls.state["generating"],
            cls.state["generating"],
            set(fn_name_token)
        )
        table.add_transition(
            cls.state["generating"],
            cls.state["stoping"],
            set(model.encode(";"))
        )
        return table


class GrammarFactory:
    @classmethod
    def create(cls, model: Model, fn_list: list[FnDef]) -> fsm.Grammar:
        return fsm.Grammar(
            TransitionTableFactory.create(model, fn_list),
            TransitionTableFactory.state["generating"],
            TransitionTableFactory.state["stoping"]
        )
