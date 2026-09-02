"""Satisfy an import that a recording path never calls.

Hosts drag in machinery no arm of a film uses: openpi's policy loader
imports its training data loader, which imports a `lerobot` layout that has
since moved; lerobot's policy factory imports every policy family; a
transitive `flash_attn` is built against a different torch. One of those
being absent should not decide whether a model can be timed.

The rule is always the same: **satisfy the import, refuse the call**, and
say in the receipt what was stubbed. A silently wrong path is far worse
than a missing one, so every attribute a stub hands out raises when used.
"""

from __future__ import annotations

import importlib.machinery
import sys
import types

__all__ = ["stub_modules"]


class _Refuses:
    _why = "stubbed"

    def __init__(self, *_a, **_k):
        raise RuntimeError(type(self)._why)

    def __call__(self, *_a, **_k):
        raise RuntimeError(type(self)._why)


class _StubModule(types.ModuleType):
    """A module whose every name is a placeholder that refuses to run."""

    def __init__(self, name: str, why: str):
        super().__init__(name)
        self.__path__: list[str] = []
        self.__file__ = f"<stub {name}>"
        self._why = why
        self._made: dict[str, type] = {}

    def __getattr__(self, name: str):
        # dunders must miss, or `inspect` and the import machinery will take
        # the placeholder for a real `__file__`, `__all__` or `__loader__`
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        if name not in self._made:
            self._made[name] = type(name, (_Refuses,), {"_why": self._why})
        return self._made[name]


def stub_modules(names, why: str) -> list[str]:
    """Install refusing stubs for `names`, skipping any that really import.

    Parents come first and each child is attached to its parent, because
    `import a.b.c` resolves `c` as an attribute of `a.b` — a stub that is
    only in `sys.modules` satisfies the module lookup and then fails the
    attribute one, which reads as a stranger error than the absence it
    replaced.

    Returns the ones that were stubbed, for the run's receipt.
    """
    made = []
    for name in sorted(set(names), key=lambda n: n.count(".")):
        if name in sys.modules:
            continue
        parent, _, leaf = name.rpartition(".")
        if not (parent and isinstance(sys.modules.get(parent), _StubModule)):
            try:
                __import__(name)
                continue
            except Exception:                               # noqa: BLE001
                pass
        module = _StubModule(name, why)
        module.__spec__ = importlib.machinery.ModuleSpec(
            name, loader=None, is_package=True)
        sys.modules[name] = module
        if parent in sys.modules:
            setattr(sys.modules[parent], leaf, module)
        made.append(name)
    return made
