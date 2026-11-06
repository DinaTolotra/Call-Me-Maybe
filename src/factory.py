from string import digits
from functools import cache
from typing import Literal

from src.llm_model import Model
from src.model import FnDef, Prompt
import src.fsm as fsm


state = {
    "stoping": fsm.State("stoping"),
    "generating": fsm.State("generating"),
    "sign": fsm.State("sign"),
    "digit": fsm.State("digit"),
    "decimal": fsm.State("decimal"),
    "fraction_digit": fsm.State("fraction_digit"),
    "opening_quote": fsm.State("opening_quote"),
    "closing_quote": fsm.State("closing_quote"),
    "non-quote": fsm.State("non-quote"),
    "boolean_start": fsm.State("boolean_start"),
    "boolean_end": fsm.State("boolean_end"),
}


class TransitionTableFactory:
    @classmethod
    def create_fn_name_table(
            cls,
            model: Model,
            fn_list: list[FnDef]
            ) -> fsm.TransitionTable:
        table = fsm.TransitionTable()
        fn_name_token: list[int] = []
        fn_list.append(FnDef(
            name="undefined",
            description="None of the functions listed",
            parameters={},
            returns={},
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
    @cache
    def create_number_table(cls, model: Model) -> fsm.TransitionTable:
        table = fsm.TransitionTable()
        digit_token: list[int] = []
        for d in digits:
            token = model.encode(d)
            digit_token.extend(token)
        digit_tokens = set(digit_token)
        table.add_transition(
            state["sign"],
            state["digit"],
            digit_tokens
        )
        table.add_transition(
            state["digit"],
            state["digit"],
            digit_tokens
        )
        table.add_transition(
            state["digit"],
            state["decimal"],
            set(model.encode("."))
        )
        table.add_transition(
            state["digit"],
            state["stoping"],
            set(model.encode(";"))
        )
        table.add_transition(
            state["decimal"],
            state["fraction_digit"],
            digit_tokens
        )
        table.add_transition(
            state["fraction_digit"],
            state["fraction_digit"],
            digit_tokens
        )
        table.add_transition(
            state["fraction_digit"],
            state["stoping"],
            set(model.encode(";"))
        )
        table.add_transition(
            state["generating"],
            state["sign"],
            set(model.encode("-"))
        )
        table.add_transition(
            state["generating"],
            state["digit"],
            digit_tokens
        )
        return table

    @classmethod
    @cache
    def create_string_table(cls, model: Model) -> fsm.TransitionTable:
        table = fsm.TransitionTable()
        non_quote_token: list[int] = []
        if model.vocab_size == 0:
            raise RuntimeError("uknown vocabulary size")
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

    @classmethod
    @cache
    def create_boolean_table(cls, model: Model) -> fsm.TransitionTable:
        table = fsm.TransitionTable()
        prefix_states: dict[tuple[int, ...], fsm.State] = {}
        sequences = {
            tuple(model.encode(literal + ";"))
            for literal in ("true", "false")
        }
        for sequence in sequences:
            prefix: tuple[int, ...] = ()
            source = state["boolean_start"]
            for index, token in enumerate(sequence):
                prefix += (token,)
                if index == len(sequence) - 1:
                    destination = state["boolean_end"]
                else:
                    destination = prefix_states.get(prefix)
                    if destination is None:
                        destination = fsm.State(
                            f"boolean_prefix_{len(prefix_states)}"
                        )
                        prefix_states[prefix] = destination
                table.add_transition(source, destination, {token})
                source = destination
        return table


class GrammarFactory:
    @classmethod
    def create_fn_name_grammar(
            cls,
            model: Model,
            fn_list: list[FnDef]
            ) -> fsm.Grammar:
        return fsm.Grammar(
            TransitionTableFactory.create_fn_name_table(model, fn_list),
            state["generating"], state["stoping"]
        )

    @classmethod
    def create_number_grammar(cls, model: Model) -> fsm.Grammar:
        return fsm.Grammar(
            TransitionTableFactory.create_number_table(model),
            state["generating"], state["stoping"]
        )

    @classmethod
    def create_string_grammar(cls, model: Model) -> fsm.Grammar:
        return fsm.Grammar(
            TransitionTableFactory.create_string_table(model),
            state["opening_quote"], state["closing_quote"]
        )

    @classmethod
    def create_boolean_grammar(cls, model: Model) -> fsm.Grammar:
        return fsm.Grammar(
            TransitionTableFactory.create_boolean_table(model),
            state["boolean_start"], state["boolean_end"]
        )


class ArgExtractionPromptFactory:
    def __init__(
            self,
            template: str, *,
            fn_name: str,
            param_name: str,
            description: str,
            instruction: Prompt,
            arg_val: dict[str, str | float | bool],
            param_list: dict[str, dict[Literal["type"], Literal["string", "number", "boolean"]]]
            ) -> None:
        self._template = template
        self.fn_name = fn_name
        self.description = description
        self.instruction = instruction
        self.param_name = param_name
        self.arg_val_str = "\n".join([
            f"'{a}' = {arg_val[a]}" for a in arg_val
        ])
        self.param_list_str = "\n".join([
            f"'{p}' type {param_list[p]["type"]}" for p in param_list
        ])

    @property
    def prompt(self) -> str:
        try:
            return self._template.replace(
                "{fn_name}", self.fn_name
            ).replace(
                "{fn_description}", self.description
            ).replace(
                "{parameter_list}", self.param_list_str
            ).replace(
                "{instruction}", self.instruction.prompt
            ).replace(
                "{previous_arguments}", self.arg_val_str
            ).replace(
                "{param_name}", self.param_name
            )
        except Exception:
            raise RuntimeError(
                "Failed to create an Argument Extraction Prompt"
            )


class FunctionPromptFactory:
    def __init__(
            self,
            template: str,
            instruction: str,
            fn_list: list[FnDef]
            ) -> None:
        self._template = template
        self.instruction = instruction
        self.fn_list_str = "\n".join([
            f"{fn.name}; {fn.description}" for fn in fn_list
        ])

    @property
    def prompt(self) -> str:
        try:
            return self._template.replace(
                "{instruction}", self.instruction
            ).replace(
                "{function_list}", self.fn_list_str
            )
        except Exception:
            raise RuntimeError(
                "Failed to create a Function Prompt:",
            )
