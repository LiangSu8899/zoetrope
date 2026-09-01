"""A `flash_attn` that imports but refuses to run.

The board's `flash_attn` extension is built against the container's own
torch and does not load under the torch the kernel artifacts need. It is
pulled in transitively — `lerobot.policies.__init__` imports the policy
factory, which imports xVLA, which imports Florence-2, which imports
flash_attn — by policies no arm of this film ever instantiates.

So the stub satisfies the import and fails loudly at the first call. A
silently wrong attention would be far worse than a missing one; nothing
in the recorded path may reach these.
"""

import importlib.machinery
import sys
import types


def install_if_broken() -> str:
    try:
        import flash_attn  # noqa: F401
        return "real"
    except Exception as exc:  # noqa: BLE001
        reason = f"{type(exc).__name__}: {str(exc)[:80]}"

    def refuse(*_args, **_kwargs):
        raise RuntimeError(
            "flash_attn is stubbed on this board: the installed extension "
            "does not load under this torch. No arm of this film uses it; "
            "reaching this call means the recorded path changed.")

    root = types.ModuleType("flash_attn")
    root.__path__ = []
    for name in ("flash_attn_func", "flash_attn_varlen_func",
                 "flash_attn_qkvpacked_func", "flash_attn_kvpacked_func"):
        setattr(root, name, refuse)
    root.__version__ = "0.0.0+stub"

    padding = types.ModuleType("flash_attn.bert_padding")
    for name in ("index_first_axis", "pad_input", "unpad_input"):
        setattr(padding, name, refuse)

    layers = types.ModuleType("flash_attn.layers")
    layers.__path__ = []
    rotary = types.ModuleType("flash_attn.layers.rotary")
    rotary.apply_rotary_emb = refuse

    modules = {"flash_attn": root,
               "flash_attn.bert_padding": padding,
               "flash_attn.layers": layers,
               "flash_attn.layers.rotary": rotary}
    for name, module in modules.items():
        # importlib.util.find_spec() is called on this package during
        # transformers' optional-dependency probing, and a module without
        # a spec makes that raise instead of answering "absent"
        module.__spec__ = importlib.machinery.ModuleSpec(
            name, loader=None, is_package=hasattr(module, "__path__"))
    sys.modules.update(modules)
    _teach_transformers()
    return f"stubbed ({reason})"


def _teach_transformers() -> None:
    """Tell transformers the stub is not an installed distribution.

    `is_flash_attn_2_available()` indexes a mapping built from installed
    distributions, keyed by importable name. A module that imports without
    having been installed is a key it does not have, so the probe raises
    `KeyError` instead of answering "absent".

    The entry has to name a distribution and it must not be the real one:
    an empty list trips `distributions[0]`, and `flash-attn` would claim
    the stub is the genuine article. A name nothing has installed makes the
    version lookup miss, which is the branch that answers "absent".
    """
    try:
        from transformers.utils import import_utils
    except Exception:                                       # noqa: BLE001
        return
    mapping = getattr(import_utils, "PACKAGE_DISTRIBUTION_MAPPING", None)
    if isinstance(mapping, dict):
        mapping.setdefault("flash_attn", ["flash-attn-stub"])


def probe() -> str:
    """Report whether the real extension loads. Installs nothing.

    A host that guards its own import (`try: import flash_attn / except
    ImportError: sdpa`) must be allowed to take that branch; stubbing the
    module underneath it turns a clean fallback into a hard failure at
    the first attention call.
    """
    try:
        import flash_attn  # noqa: F401
        return f"real ({flash_attn.__version__})"
    except Exception as exc:  # noqa: BLE001
        return (f"absent, host falls back: {type(exc).__name__}: "
                f"{str(exc)[:80]}")


def remove() -> None:
    """Drop the stub again, leaving the import as absent as it really is.

    For a host that only needs the module to survive a module-scope
    import chain: install, import, remove. Anything probing afterwards —
    transformers' `is_flash_attn_2_available()`, for one — then gets the
    truthful answer instead of a module that refuses when called.
    """
    for name in [n for n in sys.modules
                 if n == "flash_attn" or n.startswith("flash_attn.")]:
        if getattr(sys.modules[name], "__version__", "") == "0.0.0+stub" \
                or name.startswith("flash_attn."):
            del sys.modules[name]
