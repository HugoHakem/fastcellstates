"""
The generative model block: the single owner of fastcellstates' modelling
assumptions (supp. info §A1).  ``DirichletMultinomial`` is the default and,
today, only model; see ``base.Model`` for the full slot table.
"""

from .base import Model
from .dirichlet_multinomial import DirichletMultinomial
from .phi import global_phi

__all__ = ["DirichletMultinomial", "Model", "global_phi"]
