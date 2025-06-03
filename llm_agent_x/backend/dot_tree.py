class DotTree:
    def __init__(self, value=None):
        self._children = {}
        self._value = value

    def update(self, key_path, value):
        parts = key_path.split('.')
        node = self
        for part in parts[:-1]:
            if part not in node._children or not isinstance(node._children[part], DotTree):
                node._children[part] = DotTree()
            node = node._children[part]
        node._children[parts[-1]] = DotTree(value)

    def __getattr__(self, name):
        if name in self._children:
            return self._children[name]
        raise AttributeError(f"'DotTree' object has no attribute '{name}'")

    def __repr__(self):
        return f"DotTree(value={self._value}, children={list(self._children)})"

    @property
    def value(self):
        return self._value

    def resolve(self, key_path):
        """Optional: manually resolve a path, falling back to parents if needed."""
        parts = key_path.split('.')
        node = self
        for part in parts:
            if part in node._children:
                node = node._children[part]
            else:
                return None
        return node._value

    def __getitem__(self, key):
        return self.resolve(key)