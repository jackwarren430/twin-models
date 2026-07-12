"""twin.models — the shared frozen base + LoRA adapters.

``TwinBase`` and ``Adapters`` (with ``tree_global_norm`` / ``tree_l2_distance``)
live in ``base``/``adapters``, which import mlx at module top — they are the
Apple-silicon implementation. The backend-neutral rollout value types
(``GenResult`` / ``ReactResult`` / ``assemble_react``) live in ``types`` and have
NO framework dependency.

Those mlx-bound symbols are imported LAZILY (PEP 562 ``__getattr__``) so that
merely importing a submodule of this package — e.g. ``from twin.models.types
import GenResult`` on the torch backend path — does not drag mlx in. Importing
``twin.models`` runs this ``__init__``; keeping it mlx-free is what lets the DGX
Spark / torch backend run on a host with no working mlx (see DGX_SPARK.md).
``from twin.models import TwinBase`` still works on Apple silicon — the lazy
lookup imports ``base`` on first access.
"""

from twin.models.types import GenResult, ReactResult, assemble_react

# name -> submodule that defines it (imported on first attribute access).
_LAZY = {
    "TwinBase": "twin.models.base",
    "Adapters": "twin.models.adapters",
    "tree_global_norm": "twin.models.adapters",
    "tree_l2_distance": "twin.models.adapters",
}


def __getattr__(name: str):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module), name)


__all__ = [
    "TwinBase",
    "Adapters",
    "GenResult",
    "ReactResult",
    "assemble_react",
    "tree_global_norm",
    "tree_l2_distance",
]
