from typing import Annotated
import pydantic as pd


class FnDef(pd.BaseModel):
    name: Annotated[
        str,
        pd.Field(min_length=3, pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$")
    ]
    description: Annotated[
        str, pd.Field(min_length=1)
    ]
    parameters: Annotated[
        dict[
            Annotated[
                str,
                pd.Field(min_length=1, pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$")
            ],
            dict[str, str]
        ],
        pd.Field()
    ]
    returns: Annotated[
        dict[str, str],
        pd.Field(max_length=1)
    ]

    def __str__(self) -> str:
        param = ", ".join([f"{self.parameters[name]['type']} '{name}'" for name in self.parameters])
        return f"{self.name} with {param}: {self.description}"

    @staticmethod
    def parse_json(file: str) -> list["FnDef"]:
        try:
            with open(file, mode="r") as f:
                adapter = pd.TypeAdapter(list[FnDef])
                return adapter.validate_json(f.read())
        except pd.ValidationError as error:
            print("ERROR: ", error)
        except Exception as error:
            print("ERROR: ", error)
        return []


class Prompt(pd.BaseModel):
    prompt: Annotated[str, pd.Field(min_length=1)]

    @staticmethod
    def parse_json(file: str) -> list["Prompt"]:
        try:
            with open(file, mode="r") as f:
                adapter = pd.TypeAdapter(list[Prompt])
                return adapter.validate_json(f.read())
        except pd.ValidationError as error:
            print("ERROR: ", error)
        except Exception as error:
            print("ERROR: ", error)
        return []


class Template(pd.BaseModel):
    name: Annotated[str, pd.Field(min_length=1)]
    value: Annotated[str, pd.Field(min_length=1)]

    @staticmethod
    def parse_json(file: str) -> dict[str, "Template"]:
        try:
            with open(file, mode="r") as f:
                adapter = pd.TypeAdapter(dict[str, Template])
                return adapter.validate_json(f.read())
        except pd.ValidationError as error:
            print("ERROR: ", error)
        except Exception as error:
            print("ERROR: ", error)
        return {}
