"""LoRA vs. full fine-tuning for MindEye2 subject adaptation."""

__version__ = "0.1.0"

# The public names below are imported on first use (PEP 562), not when the package
# loads. Importing them eagerly pulled in torch via `lora.py`, so even torch-free modules
# such as `stats` and `trials` could not be imported -- or their tests run -- on a machine
# without torch. `from mindeye_lora import load_config` works exactly as before; torch is
# simply imported at that moment instead of at `import mindeye_lora`.
_EXPORTS = {
    "ArmConfig": ".config",
    "ExperimentConfig": ".config",
    "load_config": ".config",
    "Workspace": ".env",
    "get_workspace": ".env",
    "setup_environment": ".env",
    "LoRAConfig": ".lora",
    "LoRALinear": ".lora",
    "apply_lora": ".lora",
    "merge_lora": ".lora",
    "set_trainable": ".lora",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module, __name__), name)
    globals()[name] = value   # cache, so later lookups skip this function
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
