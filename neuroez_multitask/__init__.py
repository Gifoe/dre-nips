"""PGC-SEEG multitask components.

Internal Task1 semantics follow the existing B0 baseline:
1 = NEZ, 0 = EZ. Reported Task1 probabilities use P(EZ)=1-P(NEZ).
"""

from .model import PGCSEEGModel

__all__ = ["PGCSEEGModel"]
