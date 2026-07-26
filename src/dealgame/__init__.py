"""dealgame: imperfect-information deal-making games on OpenSpiel.

Angle A of the imperfect-information M&A/PE research program. The ``DealGame``
layer factors the parts shared by every deal game (private-type drawing, correct
information-set construction, and a zero-sum / general-sum payoff contract) so
that concrete games (takeover auctions now, alternating-offer bargaining later)
are thin subclasses rather than rebuilds.
"""

from dealgame.takeover import (
    TakeoverAuctionGame,
    TakeoverAuctionState,
    register_takeover_auction,
)
from dealgame.private_value import (
    PrivateValueAuctionGame,
    PrivateValueAuctionState,
    register_private_value_auction,
)
from dealgame.sequential_takeover import (
    SequentialTakeoverGame,
    SequentialTakeoverState,
    register_sequential_takeover,
)

__all__ = [
    "TakeoverAuctionGame",
    "TakeoverAuctionState",
    "register_takeover_auction",
    "PrivateValueAuctionGame",
    "PrivateValueAuctionState",
    "register_private_value_auction",
    "SequentialTakeoverGame",
    "SequentialTakeoverState",
    "register_sequential_takeover",
]
