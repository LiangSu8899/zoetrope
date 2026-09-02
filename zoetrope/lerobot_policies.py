"""Import `lerobot.policies` when a policy this film never runs cannot.

`lerobot.policies.__init__` imports every policy family, and its factory
imports every configuration. So one family that does not load — a config
dataclass written before a `transformers` change, an extension built against
a different torch — takes down the import of the one policy an arm actually
uses.

The rule is the same as the `flash_attn` stub: satisfy the import, refuse the
call. A family that gets stubbed is named in the return value and lands in
the arm's receipt, so "this ran against a stub" is never a silent condition.
The family an arm asked for is never stubbed; if that one is broken, the
error is the answer.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import sys
import types

__all__ = ["import_policies"]


class _Refuses:
    """Stands in for a class that was not importable."""

    _family = "?"

    def __init__(self, *_a, **_k):
        raise RuntimeError(
            f"lerobot.policies.{type(self)._family} is stubbed in this "
            f"process because it did not import; no arm of this film uses "
            f"it. Reaching this means the recorded path changed.")


class _StubModule(types.ModuleType):
    def __init__(self, name: str, family: str):
        super().__init__(name)
        self.__path__: list[str] = []
        self._family = family
        self._made: dict[str, type] = {}

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        if name not in self._made:
            self._made[name] = type(name, (_Refuses,),
                                    {"_family": self._family})
        return self._made[name]


class _StubLoader(importlib.abc.Loader):
    def __init__(self, family: str):
        self.family = family

    def create_module(self, spec):
        return _StubModule(spec.name, self.family)

    def exec_module(self, module):
        return None


class _StubFinder(importlib.abc.MetaPathFinder):
    """Resolves `lerobot.policies.<family>` and anything under it."""

    def __init__(self):
        self.families: set[str] = set()

    def find_spec(self, fullname, path=None, target=None):
        parts = fullname.split(".")
        if (len(parts) >= 3 and parts[0] == "lerobot"
                and parts[1] == "policies" and parts[2] in self.families):
            return importlib.machinery.ModuleSpec(
                fullname, _StubLoader(parts[2]), is_package=True)
        return None


_FINDER = _StubFinder()


def _family_in(exc: BaseException, root: str = "policies") -> str | None:
    """The policy family whose own file raised, from the traceback."""
    tb, found = exc.__traceback__, None
    while tb is not None:
        parts = tb.tb_frame.f_code.co_filename.replace("\\", "/").split("/")
        if root in parts:
            i = parts.index(root)
            if i + 2 < len(parts):          # <root>/<family>/<file>.py
                found = parts[i + 1]
        tb = tb.tb_next
    return found


def import_policies(*, need: str, attempts: int = 8) -> dict[str, str]:
    """Import `lerobot.policies`; return the families that had to be stubbed.

    `need` is the family this arm is about to instantiate. It is never
    stubbed — a failure there is a real failure and is raised.
    """
    if _FINDER not in sys.meta_path:
        sys.meta_path.insert(0, _FINDER)
    stubbed: dict[str, str] = {}
    for _ in range(attempts):
        try:
            importlib.import_module("lerobot.policies")
            return stubbed
        except Exception as exc:                            # noqa: BLE001
            family = _family_in(exc)
            if family is None or family == need or family in stubbed:
                raise
            stubbed[family] = f"{type(exc).__name__}: {str(exc)[:140]}"
            _FINDER.families.add(family)
            # the failed module is already out of sys.modules; the ones that
            # did load stay, so a retry does not re-run their side effects
            # (lerobot's configs register themselves with draccus by name,
            # and a second registration is an error)
    raise RuntimeError(
        f"lerobot.policies still will not import after stubbing {stubbed}")
