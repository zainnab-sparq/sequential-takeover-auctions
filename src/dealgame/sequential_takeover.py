"""Sequential (multi-round ascending) common-value takeover auction.

Story
-----
The sealed auction in :mod:`dealgame.takeover` has each bidder submit one bid,
blind to the rival. Real takeover contests are not sealed: bids arrive in rounds,
each side watches the other, and a bidder can *jump* well above the standing bid
to preempt a rival from ever showing up. That preemptive channel is exactly what
the sealed game cannot express, and it is where the classical literature says a
toehold earns its keep (Fishman 1988 preemption; Bulow--Huang--Klemperer 1999
aggressiveness). This game restores it.

Protocol
--------
Chance draws the common value ``W`` and then ``num_signals`` (k) i.i.d. noisy
signals of ``W`` for each bidder, exactly as in the sealed game. Bidding is then an
ascending auction run for at most ``num_rounds`` (R) rounds. Bidders alternate,
bidder 0 first, so each gets R turns and the auction is at most 2R bidding
decisions long. On its turn the active bidder either

* **raises** to any grid level strictly above the standing bid (a jump bid: it need
  not be the minimum increment, which is what makes preemption expressible), or
* **passes**. Passing while the rival holds the standing bid concedes: the rival
  wins at that bid. Passing when no bid stands is a wait; if both bidders wait, the
  auction ends with no sale and both earn zero.

If the round limit is reached with a bid standing, that bidder wins at it.

Because every raise must strictly exceed the standing bid, an auction can hold at
most ``num_bids`` raises in total. So the bid grid, not ``num_rounds``, is the
binding constraint unless ``num_bids >= 2 * num_rounds``; below that the extra
rounds are unreachable and the game tree stops growing.

That is what ``bid_step`` is for. It sets the money value of one bid index, so the
grid spans ``[0, (num_bids - 1) * bid_step]``. Leaving it at 1 makes the amount equal
the index (the sealed game's convention), but then a grid wide enough for many rounds
runs far past the highest common value, and every one of those levels is dominated: a
long contest becomes unrepresentable. Shrinking ``bid_step`` instead refines the grid
over a fixed, economically meaningful price range, which is what a genuinely
multi-round contest needs (and doubles as the knob for a grid-refinement robustness
check).

Information
-----------
Signals stay private; **the bid history is public**. The information state is the
bidder's own k signals plus the whole public sequence of bids and passes, not just
the standing bid, because the *path* is what carries information: a rival that
jumped straight to a high level says something different about its signal than one
that crawled there. Keeping the path is also what gives the game perfect recall.

Payoffs are scored zero-sum as the profit difference (see :mod:`dealgame.base`);
``raw_profits`` exposes the undistorted own-profit payoffs used for the economics.
"""

from __future__ import annotations

import numpy as np
import pyspiel

from dealgame.base import info_string, zero_sum_from_profits

_DEFAULT_NUM_VALUES = 3
_DEFAULT_NUM_BIDS = 4
_DEFAULT_NOISE = 0.5
_DEFAULT_NUM_SIGNALS = 1
_DEFAULT_NUM_ROUNDS = 3  # turns per bidder; >=3 is where a toehold compounds
_DEFAULT_BID_STEP = 1.0  # money per bid index; 1.0 makes the amount equal the index

_PHASE_DRAW_VALUE = 0
_PHASE_DRAW_SIGNALS = 1
_PHASE_BID = 2

_GAME_TYPE = pyspiel.GameType(
    short_name="dealgame_sequential_takeover",
    long_name="Sequential Common-Value Takeover Auction",
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
        "num_rounds": _DEFAULT_NUM_ROUNDS,
        "bid_step": _DEFAULT_BID_STEP,
    },
)


def _make_game_info(num_values: int, num_bids: int, num_signals: int,
                    num_rounds: int, bid_step: float) -> pyspiel.GameInfo:
    # Bounds are loose but valid: own profit is at most the highest value, and at
    # worst a bidder overpays by the highest bid level, so the profit difference is
    # bounded by their sum.
    bound = float(num_values) + float(num_bids - 1) * bid_step
    return pyspiel.GameInfo(
        num_distinct_actions=max(num_values, num_bids + 1),  # +1 for PASS
        max_chance_outcomes=num_values,
        num_players=2,
        min_utility=-bound,
        max_utility=bound,
        utility_sum=0.0,
        max_game_length=1 + 2 * num_signals + 2 * num_rounds,
    )


class SequentialTakeoverGame(pyspiel.Game):
    """OpenSpiel game object for the sequential common-value takeover auction."""

    def __init__(self, params=None):
        params = params or {}
        self.num_values = int(params.get("num_values", _DEFAULT_NUM_VALUES))
        self.num_bids = int(params.get("num_bids", _DEFAULT_NUM_BIDS))
        self.signal_noise_0 = float(params.get("signal_noise_0", _DEFAULT_NOISE))
        self.signal_noise_1 = float(params.get("signal_noise_1", _DEFAULT_NOISE))
        self.toehold = float(params.get("toehold", 0.0))
        self.num_signals = int(params.get("num_signals", _DEFAULT_NUM_SIGNALS))
        self.num_rounds = int(params.get("num_rounds", _DEFAULT_NUM_ROUNDS))
        self.bid_step = float(params.get("bid_step", _DEFAULT_BID_STEP))
        super().__init__(
            _GAME_TYPE,
            _make_game_info(self.num_values, self.num_bids, self.num_signals,
                            self.num_rounds, self.bid_step),
            params)
        self.value_grid = [i + 1 for i in range(self.num_values)]
        self.bid_grid = [i * self.bid_step for i in range(self.num_bids)]
        self.pass_action = self.num_bids

    def new_initial_state(self):
        return SequentialTakeoverState(self)

    def make_py_observer(self, iig_obs_type=None, params=None):
        return SequentialTakeoverObserver(self.num_values, self.num_signals,
                                          self.num_bids, self.num_rounds)


class SequentialTakeoverState(pyspiel.State):
    """A single play-through of the sequential takeover auction."""

    def __init__(self, game: SequentialTakeoverGame):
        super().__init__(game)
        self._num_values = game.num_values
        self._num_bids = game.num_bids
        self._num_signals = game.num_signals
        self._num_rounds = game.num_rounds
        self._noise = (game.signal_noise_0, game.signal_noise_1)
        self._toehold = game.toehold
        self._value_grid = list(game.value_grid)
        self._bid_grid = list(game.bid_grid)
        self._pass_action = game.pass_action
        self._max_turns = 2 * game.num_rounds

        self._phase = _PHASE_DRAW_VALUE
        self._w = None
        self._signals = [[None] * self._num_signals,
                         [None] * self._num_signals]
        self._sig_player = 0
        self._sig_idx = 0

        self._turn = 0                # bidding decisions taken so far
        self._standing_bid = None     # bid index currently leading
        self._standing_bidder = None
        self._history_tokens = []     # public: one token per bidding decision
        self._game_over = False

    # --- core OpenSpiel API -------------------------------------------------

    def current_player(self):
        if self._game_over:
            return pyspiel.PlayerId.TERMINAL
        if self._phase in (_PHASE_DRAW_VALUE, _PHASE_DRAW_SIGNALS):
            return pyspiel.PlayerId.CHANCE
        return self._turn % 2

    def _legal_actions(self, player):
        if self._phase != _PHASE_BID:
            return []
        lowest_raise = 0 if self._standing_bid is None else self._standing_bid + 1
        return list(range(lowest_raise, self._num_bids)) + [self._pass_action]

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
            return
        if self._phase == _PHASE_DRAW_SIGNALS:
            self._signals[self._sig_player][self._sig_idx] = action
            self._sig_idx += 1
            if self._sig_idx >= self._num_signals:
                self._sig_idx = 0
                self._sig_player += 1
                if self._sig_player >= 2:
                    self._phase = _PHASE_BID
            return

        player = self._turn % 2
        if action == self._pass_action:
            self._history_tokens.append("p")
            # Conceding to a standing rival bid ends the contest; waiting when no bid
            # stands does not, but if both bidders wait the auction dies.
            if self._standing_bidder is not None:
                self._game_over = True
            elif self._turn >= 1:
                self._game_over = True
        else:
            self._history_tokens.append(str(action))
            self._standing_bid = action
            self._standing_bidder = player
        self._turn += 1
        if self._turn >= self._max_turns:
            self._game_over = True

    def _action_to_string(self, player, action):
        if player == pyspiel.PlayerId.CHANCE:
            if self._phase == _PHASE_DRAW_VALUE:
                return f"W={self._value_grid[action]}"
            return f"signal={action}"
        if action == self._pass_action:
            return "pass"
        return f"bid={self._bid_grid[action]}"

    def is_terminal(self):
        return self._game_over

    def standing_bid(self):
        """Index of the bid currently leading, or ``None`` if no bid stands."""
        return self._standing_bid

    def auction_outcome(self):
        """``(winner, price)``; ``winner`` is ``None`` when the auction died."""
        if self._standing_bidder is None:
            return None, 0.0
        return self._standing_bidder, float(self._bid_grid[self._standing_bid])

    def raw_profits(self):
        """Each bidder's own (general-sum) profit, before zero-sum relativization.

        Read the economics off this, not off ``returns``: the zero-sum rendering
        subtracts the rival's endogenous profit and so injects a rivalry term the
        underlying auction does not have.
        """
        if not self._game_over or self._standing_bidder is None:
            return [0.0, 0.0]  # no sale
        w_val = self._value_grid[self._w]
        price = float(self._bid_grid[self._standing_bid])
        theta = self._toehold
        profit = [0.0, 0.0]
        if self._standing_bidder == 0:
            # The toehold holder buys only the fraction it does not already own.
            profit[0] = w_val - (1.0 - theta) * price
        else:
            profit[1] = w_val - price
            # The toehold holder loses but sells its stake into the winning price.
            profit[0] = theta * price
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
        sigs = self._signals[player]
        if self._num_signals == 1:
            private = {"sig": sigs[0]}
        else:
            private = {f"sig{i}": sigs[i] for i in range(self._num_signals)}
        # The full public path (not just the standing bid): jump size is a signal,
        # and keeping the path is what makes recall perfect.
        history = ",".join(self._history_tokens) if self._history_tokens else "-"
        return info_string(player, private, {"hist": history})

    def observation_string(self, player=None):
        return self.information_state_string(player)

    def __str__(self):
        return (
            f"w={self._w} signals={self._signals} hist={self._history_tokens} "
            f"standing={self._standing_bid}@{self._standing_bidder} "
            f"over={self._game_over}"
        )


class SequentialTakeoverObserver:
    """Observer producing an information-state tensor + string for each bidder.

    The tensor is the queried bidder's own k signals (k one-hots) concatenated with
    the public bid path: one ``num_bids + 1`` one-hot per bidding turn, the extra
    slot being "passed". Turns not yet played stay all-zero. Encoding the path
    rather than the standing bid keeps the tensor perfect-recall, so a function
    approximator sees everything the tabular information state does.
    """

    def __init__(self, num_values: int, num_signals: int, num_bids: int,
                 num_rounds: int):
        self._num_values = num_values
        self._num_signals = num_signals
        self._num_bids = num_bids
        self._turn_width = num_bids + 1
        self._max_turns = 2 * num_rounds
        self._signal_width = num_values * num_signals
        self.tensor = np.zeros(
            self._signal_width + self._max_turns * self._turn_width, np.float32)
        self.dict = {
            "signal": self.tensor[:self._signal_width],
            "bid_history": self.tensor[self._signal_width:],
        }

    def set_from(self, state, player):
        self.tensor.fill(0.0)
        for i, sig in enumerate(state._signals[player]):
            if sig is not None:
                self.tensor[i * self._num_values + sig] = 1.0
        for turn, token in enumerate(state._history_tokens):
            slot = self._num_bids if token == "p" else int(token)
            offset = self._signal_width + turn * self._turn_width
            self.tensor[offset + slot] = 1.0

    def string_from(self, state, player):
        return state.information_state_string(player)


def register_sequential_takeover():
    """Register the game with OpenSpiel (idempotent)."""
    if _GAME_TYPE.short_name not in pyspiel.registered_names():
        pyspiel.register_game(_GAME_TYPE, SequentialTakeoverGame)


register_sequential_takeover()
