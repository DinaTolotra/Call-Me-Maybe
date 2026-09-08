import llm_sdk as sdk


class Model:
    def __init__(self) -> None:
        self._model: sdk.Small_LLM_Model = sdk.Small_LLM_Model()
        self._vocab_file: str = self._model.get_path_to_vocab_file()
        self._merge_file: str = self._model.get_path_to_merges_file()
        self._tokenizer_file: str = self._model.get_path_to_tokenizer_file()

    def encode(self, text: str) -> ...:
        tensor: list[list[int]] = self._model.encode(text).tolist()
        return tensor[0]

    def decode(self, ids: list[int]) -> str:
        return self._model.decode(ids)

    def infer(self, input_ids: list[int]) -> list[float]:
        return self._model.get_logits_from_input_ids(input_ids)

    @property
    def vocab_file(self) -> str:
        return self._vocab_file

    @property
    def merge_file(self) -> str:
        return self._merge_file

    @property
    def tokenizer_file(self) -> str:
        return self._tokenizer_file
