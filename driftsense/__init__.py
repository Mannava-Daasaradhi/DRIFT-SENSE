"""DRIFT-SENSE — navigation-error recovery for periodic wafer layouts.

Package layout (see docs/INTERFACES.md §4 for ownership):

    layouts, sem_physics      Member A — synthetic DRAM/FinFET geometry + SEM forward model
    preprocess, spectral,     Member B — the localization core
    matching, periodic, decide
    rerank, viz               Member C — optional CNN re-ranker and figures

Nothing in this package imports torch at module scope. The classical localization
path must run with zero ML dependencies installed (PLAN.md Rule 1).
"""

__version__ = "0.1.0"
