"""Back-compat shim: the DM reference moved into the package.
See ``fastcellstates.model._dm_reference``.
"""

from fastcellstates.model._dm_reference import merge_delta, partition_loglik

__all__ = ["merge_delta", "partition_loglik"]
