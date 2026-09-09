class Node:
    def __init__(self, token: int):
        self.token = token
        self._children = dict[int, Node]()

    def __getitem__(self, token: int) -> "Node":
        return self._children.get(token, Node(-1))

    def has(self, token) -> bool:
        return self._children.get(token, None) is not None

    def add_child(self, token: int):
        self._children[token] = Node(token)
        return self._children[token]

    def get_token_option(self) -> set[int]:
        return set(self._children.keys())

    def get_children_list(self) -> list["Node"]:
        return list(self._children.values())

    def get_children_count(self) -> int:
        return len(self.get_children_list())


def print_tree(node: Node, depth: int) -> None:
    print(" " * depth + f"[{node.token}]")
    for child in node.get_children_list():
        print_tree(child, depth + 1)
