from typing import List, Hashable, Callable


class Recaller:
    def __init__(self, allowed_keys: List[Hashable]):
        self.fns = {key: [] for key in allowed_keys}

    def register(self, key: Hashable, fn: Callable[..., None]):
        self.fns[key].append(fn)
    
    def trigger(self, key: Hashable, *args, **kwargs):
        for fn in self.fns[key]:
            fn(*args, **kwargs)
