from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated, Any, Literal, Optional
import pydantic as pd


MAX_JSON_BYTES = 2_000_000


def _read_json(file: str, adapter: pd.TypeAdapter[Any]) -> Any:
    with open(file, mode="rb") as source:
        content = source.read(MAX_JSON_BYTES + 1)
    if len(content) > MAX_JSON_BYTES:
        raise ValueError("JSON input exceeds the maximum size")
    return adapter.validate_json(content)


class FnDef(pd.BaseModel):
    model_config = pd.ConfigDict(extra="forbid", strict=True)

    name: Annotated[
        str,
        pd.Field(min_length=3, pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$")
    ]
    description: Annotated[
        str, pd.Field(min_length=1, max_length=4096)
    ]
    parameters: Annotated[
        dict[
            Annotated[
                str,
                pd.Field(min_length=1, pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$")
            ],
            Annotated[
                dict[Literal["type"], Literal["string", "number", "boolean"]],
                pd.Field(min_length=1, max_length=1)
            ]
        ],
        pd.Field(max_length=64)
    ]
    returns: Annotated[
        dict[Literal["type"], Literal["string", "number", "boolean"]],
        pd.Field(min_length=1, max_length=1)
    ]

    def __str__(self) -> str:
        param = ", ".join([
            f"{self.parameters[name]['type']} '{name}'"
            for name in self.parameters
        ])
        return f"{self.name} with {param}: {self.description}"

    @staticmethod
    def parse_json(file: str) -> list["FnDef"]:
        adapter = pd.TypeAdapter(
            Annotated[list[FnDef], pd.Field(max_length=256)]
        )
        return _read_json(file, adapter)


class Prompt(pd.BaseModel):
    model_config = pd.ConfigDict(extra="forbid", strict=True)

    prompt: Annotated[str, pd.Field(min_length=1, max_length=16_384)]

    @staticmethod
    def parse_json(file: str) -> list["Prompt"]:
        adapter = pd.TypeAdapter(
            Annotated[list[Prompt], pd.Field(max_length=1000)]
        )
        return _read_json(file, adapter)


class Template(pd.BaseModel):
    model_config = pd.ConfigDict(extra="forbid", strict=True)

    name: Annotated[str, pd.Field(min_length=1)]
    value: Annotated[str, pd.Field(min_length=1, max_length=16_384)]

    @staticmethod
    def parse_json(file: str) -> dict[str, "Template"]:
        adapter = pd.TypeAdapter(
            Annotated[dict[str, Template], pd.Field(max_length=32)]
        )
        return _read_json(file, adapter)


class Result(pd.BaseModel):
    model_config = pd.ConfigDict(extra="forbid", strict=True)

    prompt: Annotated[str, pd.Field(min_length=1)]
    name: Annotated[
        Optional[str],
        pd.Field(min_length=3, pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$")
    ]
    parameters: Annotated[dict[str, str | float], pd.Field()]

    def add_parameter(self, name: str, value: str) -> None:
        self.parameters[name] = value

    @staticmethod
    def save_list(result_list: list["Result"], file: str) -> None:
        adapter = pd.TypeAdapter(list[Result])
        payload = adapter.dump_json(result_list, indent=4)
        destination = Path(file)
        temporary_path: str | None = None
        try:
            with NamedTemporaryFile(
                    mode="wb", dir=destination.parent, delete=False
                    ) as temporary:
                temporary_path = temporary.name
                temporary.write(payload)
                temporary.flush()
            Path(temporary_path).replace(destination)
        finally:
            if temporary_path is not None:
                Path(temporary_path).unlink(missing_ok=True)
