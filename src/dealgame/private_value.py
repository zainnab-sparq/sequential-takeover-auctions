"""Private-value first-price auction as a two-player zero-sum game.

A structurally different deal game from the common-value takeover auction: here
each bidder draws its *own* private value for the target (independent private
values) and observes it exactly. There is no common value and no winner's curse;
the strategic problem is bid *shading* against an opponent of unknown value. This
gives the benchmark a second, qualitatively different game so results are not an
artifact of the winner's-curse structure.

Sealed first-price: higher bid wins (ties to bidder 0), pays its own bid, earns
``value - bid``; the loser earns nothing. Scored zero-sum as the profit
difference.
"""

from __future__ import annotations

import numpy as np
import pyspiel

from dealgame.base import info_string, zero_sum_from_profits

_DEFAULT_NUM_VALUES = 4
_DEFAULT_NUM_BIDS = 4

_PHASE_DRAW_V0 = 0
_PHASE_DRAW_V1 = 1
_PHASE_BID0 = 2
_PHASE_BID1 = 3

_GAME_TYPE = pyspiel.GameType(
    short_name="dealgame_private_value_auction",
    long_name="Private-Value First-Price Auction",
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
    },
)


def _make_game_info(num_values: int, num_bids: int) -> pyspiel.GameInfo:
    return pyspiel.GameInfo(
        num_distinct_actions=max(num_values, num_bids),
        max_chance_outcomes=num_values,
        num_players=2,
        min_utility=float(-(num_bids - 1) - num_values),
        max_utility=float(num_values + num_values),
        utility_sum=0.0,
        max_game_length=4,
    )


class PrivateValueAuctionGame(pyspiel.Game):
    def __init__(self, params=None):
        params = params or {}
        self.num_values = int(params.get("num_values", _DEFAULT_NUM_VALUES))
        self.num_bids = int(params.get("num_bids", _DEFAULT_NUM_BIDS))
        super().__init__(_GAME_TYPE, _make_game_info(self.num_values, self.num_bids), params)
        self.value_grid = [i + 1 for i in range(self.num_values)]
        self.bid_grid = list(range(self.num_bids))

    def new_initial_state(self):
        return PrivateValueAuctionState(self)

    def make_py_observer(self, iig_obs_type=None, params=None):
        return _ValueObserver(self.num_values)


class PrivateValueAuctionState(pyspiel.State):
    def __init__(self, game: PrivateValueAuctionGame):
        super().__init__(game)
        self._num_values = game.num_values
        self._num_bids = game.num_bids
        self._value_grid = list(game.value_grid)
        self._bid_grid = list(game.bid_grid)
        self._phase = _PHASE_DRAW_V0
        self._values = [None, None]
        self._bids = [None, None]
        self._game_over = False

    def current_player(self):
        if self._game_over:
            return pyspiel.PlayerId.TERMINAL
        if self._phase in (_PHASE_DRAW_V0, _PHASE_DRAW_V1):
            return pyspiel.PlayerId.CHANCE
        if self._phase == _PHASE_BID0:
            return 0
        return 1

    def _legal_actions(self, player):
        if self._phase in (_PHASE_BID0, _PHASE_BID1):
            return list(range(self._num_bids))
        return []

    def chance_outcomes(self):
        p = 1.0 / self._num_values
        return [(i, p) for i in range(self._num_values)]

    def _apply_action(self, action):
        if self._phase == _PHASE_DRAW_V0:
            self._values[0] = action
            self._phase = _PHASE_DRAW_V1
        elif self._phase == _PHASE_DRAW_V1:
            self._values[1] = action
            self._phase = _PHASE_BID0
        elif self._phase == _PHASE_BID0:
            self._bids[0] = action
            self._phase = _PHASE_BID1
        else:
            self._bids[1] = action
            self._game_over = True

    def _action_to_string(self, player, action):
        if player == pyspiel.PlayerId.CHANCE:
            return f"v={self._value_grid[action]}"
        return f"bid={self._bid_grid[action]}"

    def is_terminal(self):
        return self._game_over

    def returns(self):
        if not self._game_over:
            return [0.0, 0.0]
        b0, b1 = self._bid_grid[self._bids[0]], self._bid_grid[self._bids[1]]
        profit = [0.0, 0.0]
        if b0 >= b1:
            profit[0] = self._value_grid[self._values[0]] - b0
        else:
            profit[1] = self._value_grid[self._values[1]] - b1
        return zero_sum_from_profits(profit[0], profit[1])

    def information_state_string(self, player=None):
        if player is None:
            player = self.current_player()
        return info_string(player, {"val": self._values[player]}, {})

    def observation_string(self, player=None):
        return self.information_state_string(player)

    def __str__(self):
        return f"values={self._values} bids={self._bids} over={self._game_over}"


class _ValueObserver:
    def __init__(self, num_values: int):
        self.tensor = np.zeros(num_values, np.float32)
        self.dict = {"value": self.tensor}

    def set_from(self, state, player):
        self.tensor.fill(0.0)
        v = state._values[player]
        if v is not None:
            self.tensor[v] = 1.0

    def string_from(self, state, player):
        return state.information_state_string(player)


def register_private_value_auction():
    if _GAME_TYPE.short_name not in pyspiel.registered_names():
        pyspiel.register_game(_GAME_TYPE, PrivateValueAuctionGame)


register_private_value_auction()
