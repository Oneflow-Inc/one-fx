
"""
Modified from https://github.com/pytorch/pytorch/blob/master/torch/multiprocessing/reductions.py
"""

class StorageWeakRef(object):
    r"""A weak reference to a Storage.

    The cdata member is a Python number containing the integer representation of
    the Storage pointer."""

    def __init__(self, storage):
        self.cdata = storage._weak_ref()
        # Save a direct reference to _free_weak_ref because the `torch` module
        # might be cleared during Python shutdown before this module is cleared.
        self._free_weak_ref = torch.Storage._free_weak_ref  # type: ignore[attr-defined]

    def expired(self):
        return torch.Storage._expired(self.cdata)  # type: ignore[attr-defined]

    def __del__(self):
        self._free_weak_ref(self.cdata)

    def __hash__(self):
        return self.cdata

    def __eq__(self, other):
        if id(self) == id(other):
            return True
        return self.cdata == other.cdata