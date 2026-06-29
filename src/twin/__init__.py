"""Twin-Models: self-play RLVR with two LoRA adapters over one frozen MLX base.

See DESIGN.md for the full specification. Sprint 1 ships:
  - twin.config       : typed config + YAML loader
  - twin.models.base  : TwinBase (load / generate / oracle / logprobs)
  - twin.models.adapters : Adapters (two LoRA trees over one shared frozen base)
  - twin.problems.schema : Problem / ProblemSuite (+ parsing & validation)
"""

__version__ = "0.1.0"
