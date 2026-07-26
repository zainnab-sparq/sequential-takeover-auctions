"""Common-value takeover auction as a two-player zero-sum imperfect-information game.

Story
-----
Two bidders (e.g. competing PE firms) bid for a target whose value ``W`` is
*common* to both but unknown: the target is worth the same ``W`` to whoever wins,
but neither bidder observes ``W`` directly. Each privately receives one or more
noisy *signals* of ``W``. They submit sealed first-price bids simultaneously
(bidder 1 does not see bidder 0's bid). The higher bid wins, pays its own bid, and
earns ``W - bid``; the loser earns nothing.

The common-value structure produces a **winner's curse**: conditional on winning,
a bidder had the higher signal, so the asset is worth less in expectation than its
signal suggested; a naive bidder who does not condition on winning overpays
systematically. The strategic response is to shade bids. ``signal_noise_0`` /
``signal_noise_1`` are the per-bidder asymmetry knobs (0 = perfect signal,
1 = pure noise); a sharper signal lets a bidder estimate ``W`` better and shade
correctly. This is a stylized first-price common-value auction
(Milgrom--Weber; Klemperer); the sealed simultaneous structure abstracts away the
*sequential* preemptive/jump bidding studied by Fishman.

``num_signals`` (k) gives each bidder k i.i.d. noisy signals of ``W`` (k independent
due-diligence estimates). An information set is the bidder's k-tuple of signals, so
the number of information sets per bidder is ``num_values**k`` and grows exponentially
with k: at k=1 the policy is tiny and tabulatable, but for moderate k it is far too
large to tabulate
and the game tree (``num_values**(2k+1) * num_bids**2``) is far too large to
enumerate, so exact CFR and exact exploitability cannot run. Function approximation
can still generalize across signal vectors (the relevant statistic is roughly their
mean), which is exactly the regime where the learning methods are meant to earn their
keep. k=1 recovers the original single-signal game.

Payoffs are scored zero-sum as the *difference* in profit (player 0 profit minus
player 1 profit), the head-to-head benchmark used throughout the seed paper, so
OpenSpiel's exact exploitability applies. As noted in :mod:`dealgame.base`, this
relativization adds a rivalry term, so the equilibrium found is that of the
zero-sum rendering, not the auction's Bayes-Nash equilibrium; ``raw_profits``
exposes the undistorted own-profit payoffs.
"""

from __future__ import annotations

import numpy as np
import pyspiel

from dealgame.base import info_string, zero_sum_from_profits

_DEFAULT_NUM_VALUES = 3   # support of the common value W
_DEFAULT_NUM_BIDS = 4     # number of discrete bid levels
_DEFAULT_NOISE = 0.5
_DEFAULT_NUM_SIGNALS = 1  # signals per bidder (k); >1 makes the game intractable

_PHASE_DRAW_VALUE = 0
_PHASE_DRAW_SIGNALS = 1
_PHASE_BID0 = 2
_PHASE_BID1 = 3

_GAME_TYPE = pyspiel.GameType(
    short_name="dealgame_takeover_auction",
    long_name="Common-Value Takeover Auction",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=2,
    min_num_players=2,
    provides_information_state_string=True,
    provides_information_state_tensor=True,
    provides_observation_string=True,
    provides_observation_tensor=True,
    parameter_specification={
        "num_values": _DEFAULT_NUM_VALUES,
        "num_bids": _DEFAULT_NUM_BIDS,
        "signal_noise_0": _DEFAULT_NOISE,
        "signal_noise_1": _DEFAULT_NOISE,
        "toehold": 0.0,
        "num_signals": _DEFAULT_NUM_SIGNALS,
    },
)


def _make_game_info(num_values: int, num_bids: int,
                    num_signals: int) -> pyspiel.GameInfo:
    max_value = num_values  # values are 1..num_values
    return pyspiel.GameInfo(
        num_distinct_actions=max(num_values, num_bids),
        max_chance_outcomes=num_values,
        num_players=2,
        min_utility=float(-(num_bids - 1) - (max_value - 1)),
        max_utility=float(max_value - 1 + max_value),
        utility_sum=0.0,
        max_game_length=3 + 2 * num_signals,
    )


class TakeoverAuctionGame(pyspiel.Game):
    """OpenSpiel game object for the common-value takeover auction."""

    def __init__(self, params=None):
        params = params or {}
        self.num_values = int(params.get("num_values", _DEFAULT_NUM_VALUES))
        self.num_bids = int(params.get("num_bids", _DEFAULT_NUM_BIDS))
        self.signal_noise_0 = float(params.get("signal_noise_0", _DEFAULT_NOISE))
        self.signal_noise_1 = float(params.get("signal_noise_1", _DEFAULT_NOISE))
        # Fraction of the target already owned by bidder 0 (a "toehold"). If a
        # rival wins, the toehold holder still collects its share of the price.
        self.toehold = float(params.get("toehold", 0.0))
        self.num_signals = int(params.get("num_signals", _DEFAULT_NUM_SIGNALS))
        super().__init__(_GAME_TYPE,
                         _make_game_info(self.num_values, self.num_bids,
                                         self.num_signals), params)
        # Common value support 1..num_values; bid levels 0..num_bids-1.
        self.value_grid = [i + 1 for i in range(self.num_values)]
        self.bid_grid = list(range(self.num_bids))

    def new_initial_state(self):
        return TakeoverAuctionState(self)

    def make_py_observer(self, iig_obs_type=None, params=None):
        return TakeoverObserver(self.num_values, self.num_signals)


class TakeoverAuctionState(pyspiel.State):
    """A single play-through of the common-value takeover auction."""

    def __init__(self, game: TakeoverAuctionGame):
        super().__init__(game)
        # Copy parameters as primitives so cloned states stay self-contained.
        self._num_values = game.num_values
        self._num_bids = game.num_bids
        self._num_signals = game.num_signals
        self._noise = (game.signal_noise_0, game.signal_noise_1)
        self._toehold = game.toehold
        self._value_grid = list(game.value_grid)
        self._bid_grid = list(game.bid_grid)
        self._phase = _PHASE_DRAW_VALUE
        self._w = None        # common value index
        # per-bidder list of k private signal indices
        self._signals = [[None] * self._num_signals,
                         [None] * self._num_signals]
        self._sig_player = 0  # which bidder's signals are being drawn
        self._sig_idx = 0     # which signal index within that bidder
        self._bids = [None, None]     # per-bidder bid index
        self._game_over = False

    # --- core OpenSpiel API -------------------------------------------------

    def current_player(self):
        if self._game_over:
            return pyspiel.PlayerId.TERMINAL
        if self._phase in (_PHASE_DRAW_VALUE, _PHASE_DRAW_SIGNALS):
            return pyspiel.PlayerId.CHANCE
        if self._phase == _PHASE_BID0:
            return 0
        return 1

    def _legal_actions(self, player):
        if self._phase in (_PHASE_BID0, _PHASE_BID1):
            return list(range(self._num_bids))
        return []

    def _signal_distribution(self, noise):
        n = self._num_values
        probs = []
        for j in range(n):
            p = noise / n
            if j == self._w:
                p += 1.0 - noise
            if p > 0.0:
                probs.append((j, p))
        return probs

    def chance_outcomes(self):
        if self._phase == _PHASE_DRAW_VALUE:
            p = 1.0 / self._num_values
            return [(i, p) for i in range(self._num_values)]
        return self._signal_distribution(self._noise[self._sig_player])

    def _apply_action(self, action):
        if self._phase == _PHASE_DRAW_VALUE:
            self._w = action
            self._phase = _PHASE_DRAW_SIGNALS
            self._sig_player = 0
            self._sig_idx = 0
        elif self._phase == _PHASE_DRAW_SIGNALS:
            self._signals[self._sig_player][self._sig_idx] = action
            self._sig_idx += 1
            if self._sig_idx >= self._num_signals:
                self._sig_idx = 0
                self._sig_player += 1
                if self._sig_player >= 2:
                    self._phase = _PHASE_BID0
        elif self._phase == _PHASE_BID0:
            self._bids[0] = action
            self._phase = _PHASE_BID1
        else:
            self._bids[1] = action
            self._game_over = True

    def _action_to_string(self, player, action):
        if player == pyspiel.PlayerId.CHANCE:
            if self._phase == _PHASE_DRAW_VALUE:
                return f"W={self._value_grid[action]}"
            return f"signal={action}"
        return f"bid={self._bid_grid[action]}"

    def is_terminal(self):
        return self._game_over

    def raw_profits(self):
        """Each bidder's own (general-sum) profit, before zero-sum relativization.

        Exposed so economic comparative statics can be measured on a bidder's own
        profit rather than the profit *difference*, which is the object the zero-sum
        rendering optimizes. The difference scoring injects a rivalry incentive that
        is absent from the underlying auction, so a clean comparative static (e.g.
        the toehold effect) should be read off own profit.
        """
        if not self._game_over:
            return [0.0, 0.0]
        w_val = self._value_grid[self._w]
        b0, b1 = self._bid_grid[self._bids[0]], self._bid_grid[self._bids[1]]
        theta = self._toehold
        profit = [0.0, 0.0]
        if b0 >= b1:  # ties go to bidder 0
            # Toehold holder buys only the (1 - theta) it does not already own.
            profit[0] = w_val - (1.0 - theta) * b0
        else:
            profit[1] = w_val - b1
            # Toehold holder loses but collects its share of the winning price.
            profit[0] = theta * b1
        return profit

    def returns(self):
        if not self._game_over:
            return [0.0, 0.0]
        profit = self.raw_profits()
        return zero_sum_from_profits(profit[0], profit[1])

    # --- information sets ---------------------------------------------------

    def information_state_string(self, player=None):
        if player is None:
            player = self.current_player()
        # Each bidder sees only its own signal(s) (bids are simultaneous: bidder 1
        # never observes bidder 0's bid). For k=1 the token is "sig" (so the string
        # is unchanged from the single-signal game); for k>1 they are "sig0".."sigk".
        sigs = self._signals[player]
        if self._num_signals == 1:
            private = {"sig": sigs[0]}
        else:
            private = {f"sig{i}": sigs[i] for i in range(self._num_signals)}
        return info_string(player, private, {})

    def observation_string(self, player=None):
        return self.information_state_string(player)

    def __str__(self):
        return (
            f"w={self._w} signals={self._signals} bids={self._bids} "
            f"over={self._game_over}"
        )


class TakeoverObserver:
    """Observer producing an information-state tensor + string for each bidder.

    Encodes only the queried bidder's own k signals (k concatenated one-hots), which
    is precisely its information set: bids are simultaneous, so nothing else is
    observed before acting.
    """

    def __init__(self, num_values: int, num_signals: int):
        self._num_values = num_values
        self._num_signals = num_signals
        self.tensor = np.zeros(num_values * num_signals, np.float32)
        self.dict = {"signal": self.tensor}

    def set_from(self, state, player):
        self.tensor.fill(0.0)
        for i, sig in enumerate(state._signals[player]):
            if sig is not None:
                self.tensor[i * self._num_values + sig] = 1.0

    def string_from(self, state, player):
        return state.information_state_string(player)


def register_takeover_auction():
    """Register the game with OpenSpiel (idempotent)."""
    if _GAME_TYPE.short_name not in pyspiel.registered_names():
        pyspiel.register_game(_GAME_TYPE, TakeoverAuctionGame)


register_takeover_auction()
