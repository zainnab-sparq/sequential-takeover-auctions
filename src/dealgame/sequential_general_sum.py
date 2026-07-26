"""Exact own-profit Bayes-Nash equilibrium for the *sequential* takeover auction.

Why this exists
---------------
Two solvers already in this package are the wrong tool here.

* The zero-sum solvers (CFR/MMD/PSRO) optimize the profit *difference*, which
  injects a rivalry term the auction does not have, so the equilibrium they find
  is not the auction's own-profit Bayes-Nash equilibrium (see :mod:`dealgame.base`).
  Read no economics off them.
* :mod:`dealgame.general_sum` does compute the own-profit equilibrium, but only
  because the *sealed* auction factorizes: terminal profit depends on nothing but
  ``(w, b0, b1)``, so the game collapses into dense tensors and no tree is needed.
  A sequential auction has no such factorization; profit depends on the whole bid
  path. There is no way around traversing the tree.

So this module runs fictitious play in **extensive form**: each round, every bidder
computes an exact own-profit best response to the rival's running average by tree
traversal, and the averages absorb those best responses. NashConv (the summed
own-profit gain from a unilateral deviation) is reported every round, so
convergence is measured rather than assumed. In practice it falls off as ~1.4/t.

Best responses in an imperfect-information game cannot be computed node by node: a
bidder picks one action per *information set*, so an action's value must be summed
over every history in that set, each weighted by the probability that chance and the
rival steer play there (the counterfactual reach). That aggregation is why values are
computed one depth at a time rather than in tree order.

Cost control
------------
Walking the game through ``pyspiel`` on every round would dominate the runtime, so
:class:`SequentialAuctionTree` walks it once and caches it as flat NumPy arrays:
children, chance probabilities and terminal profits are stored in fixed-width padded
matrices, and every round is then a handful of vectorized gathers per depth. The
cache is pinned to the real game by ``tests/test_sequential_general_sum.py``, which
also checks the best response against a brute-force search over every pure policy.
"""

from __future__ import annotations

import warnings

import numpy as np

FP_TOLERANCE = 1e-4
# NashConv falls off as ~1.4/t (measured, clean, no cycling), so reaching 1e-4 takes
# on the order of 15k rounds; the budget is set to clear that with headroom.
FP_MAX_ITERATIONS = 20000

_TERMINAL = 0
_CHANCE = 1
_DECISION = 2

Policy = list[np.ndarray]  # one probability vector per information set


class SequentialAuctionTree:
    """A pyspiel-free cache of a small sequential auction's game tree.

    Nodes are numbered in depth-first pre-order. Each information set lives at a
    single depth (its string carries the whole public bid history, and history
    length fixes the depth), which is what lets a whole information set be
    aggregated in one vectorized step.
    """

    def __init__(self, game):
        self.game = game
        self.num_bids = game.num_bids
        self.pass_action = game.pass_action
        self.first_bid_depth = 1 + 2 * game.num_signals
        # Money per bid index. With a refined grid the index is NOT the price, so any
        # statistic reported in money has to go through this.
        self.bid_amount = np.array(game.bid_grid, dtype=float)

        kind: list[int] = []
        depth: list[int] = []
        player: list[int] = []
        infoset: list[int] = []
        children: list[list[int]] = []
        chance_probs: list[list[float]] = []
        profits: list[tuple[float, float]] = []
        winner: list[int | None] = []
        price: list[float] = []
        self._raw = (kind, depth, player, infoset, children, chance_probs,
                     profits, winner, price)

        self._infoset_key: list[list[str]] = [[], []]
        self._infoset_index: list[dict[str, int]] = [{}, {}]
        self._infoset_actions: list[list[list[int]]] = [[], []]
        self._infoset_depth: list[list[int]] = [[], []]
        self._infoset_standing: list[list[bool]] = [[], []]

        self.root = self._build(game.new_initial_state(), 0)
        self._compile()
        del self._raw

    # --- construction -------------------------------------------------------

    def _new_node(self, node_kind: int, node_depth: int) -> int:
        kind, depth, player, infoset, children, chance_probs, profits, winner, price = self._raw
        node = len(kind)
        kind.append(node_kind)
        depth.append(node_depth)
        player.append(-1)
        infoset.append(-1)
        children.append([])
        chance_probs.append([])
        profits.append((0.0, 0.0))
        winner.append(None)
        price.append(0.0)
        return node

    def _build(self, state, node_depth: int) -> int:
        kind, depth, player, infoset, children, chance_probs, profits, winner, price = self._raw

        if state.is_terminal():
            node = self._new_node(_TERMINAL, node_depth)
            own = state.raw_profits()
            profits[node] = (own[0], own[1])
            winner[node], price[node] = state.auction_outcome()
            return node

        if state.is_chance_node():
            node = self._new_node(_CHANCE, node_depth)
            outcomes = state.chance_outcomes()
            chance_probs[node] = [p for _, p in outcomes]
            children[node] = [self._build(state.child(a), node_depth + 1)
                              for a, _ in outcomes]
            return node

        acting = state.current_player()
        node = self._new_node(_DECISION, node_depth)
        player[node] = acting
        legal = state.legal_actions()
        key = state.information_state_string(acting)
        index = self._infoset_index[acting].get(key)
        if index is None:
            index = len(self._infoset_key[acting])
            self._infoset_index[acting][key] = index
            self._infoset_key[acting].append(key)
            self._infoset_actions[acting].append(list(legal))
            self._infoset_depth[acting].append(node_depth)
            self._infoset_standing[acting].append(state.standing_bid() is not None)
        elif self._infoset_depth[acting][index] != node_depth:
            raise AssertionError(
                f"information set {key!r} spans two depths; the solver's depthwise "
                "aggregation assumes it cannot")
        infoset[node] = index
        children[node] = [self._build(state.child(a), node_depth + 1) for a in legal]
        return node

    def _compile(self) -> None:
        """Flatten the walked tree into padded matrices for vectorized traversal."""
        kind, depth, player, infoset, children, chance_probs, profits, winner, price = self._raw
        self.num_nodes = len(kind)
        self.max_depth = max(depth)
        self.width = max((len(c) for c in children), default=0)

        self._kind = np.array(kind)
        self._depth = np.array(depth)
        self._player = np.array(player)
        self._node_infoset = np.array(infoset)
        self._profits = np.array(profits)
        self.winner = winner
        self.price = np.array(price)

        # Padded slots point at the root and always carry zero probability. The root
        # is nobody's child, so a scatter-add into it is a no-op, which lets the
        # traversals ignore the padding entirely.
        self._child_pad = np.zeros((self.num_nodes, self.width), dtype=np.int64)
        self._chance_pad = np.zeros((self.num_nodes, self.width))
        self._valid_pad = np.zeros((self.num_nodes, self.width))
        for node, kids in enumerate(children):
            n = len(kids)
            self._child_pad[node, :n] = kids
            self._valid_pad[node, :n] = 1.0
            if kind[node] == _CHANCE:
                self._chance_pad[node, :n] = chance_probs[node]

        nodes = np.arange(self.num_nodes)
        self._terminals = nodes[self._kind == _TERMINAL]
        internal = self._kind != _TERMINAL
        self._internal_by_depth = [
            nodes[internal & (self._depth == d)] for d in range(self.max_depth + 1)]
        self._decision_nodes = [
            nodes[(self._kind == _DECISION) & (self._player == p)] for p in range(2)]
        self._decision_by_depth = [
            [nodes[(self._kind == _DECISION) & (self._player == p)
                   & (self._depth == d)] for d in range(self.max_depth + 1)]
            for p in range(2)]
        # Everything the best-responder does not control, by depth: chance plus the
        # rival's own decisions.
        self._passive_by_depth = [
            [np.concatenate([nodes[(self._kind == _CHANCE) & (self._depth == d)],
                             self._decision_by_depth[1 - p][d]])
             for d in range(self.max_depth + 1)]
            for p in range(2)]

        self._infoset_valid = []
        self._infosets_by_depth = []
        for p in range(2):
            counts = np.array([len(a) for a in self._infoset_actions[p]])
            valid = np.arange(self.width)[None, :] < counts[:, None]
            self._infoset_valid.append(valid)
            depths = np.array(self._infoset_depth[p])
            self._infosets_by_depth.append(
                [np.arange(len(counts))[depths == d] if len(counts) else np.empty(0, int)
                 for d in range(self.max_depth + 1)])

    # --- information-set accessors -----------------------------------------

    def num_infosets(self, player: int) -> int:
        return len(self._infoset_key[player])

    def infoset_actions(self, player: int, index: int) -> list[int]:
        return self._infoset_actions[player][index]

    def infoset_key(self, player: int, index: int) -> str:
        return self._infoset_key[player][index]

    def infoset_index(self, player: int, key: str) -> int:
        return self._infoset_index[player][key]

    def uniform_policy(self, player: int) -> Policy:
        return [np.full(len(actions), 1.0 / len(actions))
                for actions in self._infoset_actions[player]]

    def random_policy(self, player: int, rng: np.random.Generator) -> Policy:
        """A random full-support behavioral policy, for restarting fictitious play.

        Full support matters: a zero-probability action would make some of the rival's
        information sets unreachable, and a best response there would be arbitrary.
        """
        return [rng.dirichlet(np.ones(len(actions)))
                for actions in self._infoset_actions[player]]

    # --- vectorized traversal primitives ------------------------------------

    def _pad_policy(self, player: int, policy: Policy) -> np.ndarray:
        padded = np.zeros((self.num_infosets(player), self.width))
        for index, probs in enumerate(policy):
            padded[index, :len(probs)] = probs
        return padded

    def _action_probs(self, policy0: Policy, policy1: Policy,
                      skip_player: int | None = None) -> np.ndarray:
        """Per-node action probabilities, padded to a fixed width.

        ``skip_player`` replaces that player's probabilities with ones, which turns a
        reach computation into the *counterfactual* reach a best response needs: the
        chance that everything outside the player's control steers play to a node.
        """
        probs = self._chance_pad.copy()
        for player, policy in ((0, policy0), (1, policy1)):
            nodes = self._decision_nodes[player]
            if len(nodes) == 0:
                continue
            if player == skip_player:
                probs[nodes] = self._valid_pad[nodes]
            else:
                padded = self._pad_policy(player, policy)
                probs[nodes] = padded[self._node_infoset[nodes]]
        return probs

    def _reach(self, probs: np.ndarray) -> np.ndarray:
        """Probability of reaching each node, given per-node action probabilities."""
        reach = np.zeros(self.num_nodes)
        reach[self.root] = 1.0
        for depth in range(self.max_depth):
            nodes = self._internal_by_depth[depth]
            if len(nodes) == 0:
                continue
            contribution = reach[nodes, None] * probs[nodes]
            # Safe despite repeated padding indices: every pad points at the root and
            # contributes zero, and each real child has exactly one parent.
            reach[self._child_pad[nodes].ravel()] += contribution.ravel()
        return reach


def policy_value(tree: SequentialAuctionTree, policy0: Policy,
                 policy1: Policy) -> tuple[float, float]:
    """Expected own profit for each bidder under the given behavioral policies."""
    probs = tree._action_probs(policy0, policy1)
    values = np.zeros((tree.num_nodes, 2))
    values[tree._terminals] = tree._profits[tree._terminals]
    for depth in range(tree.max_depth - 1, -1, -1):
        nodes = tree._internal_by_depth[depth]
        if len(nodes) == 0:
            continue
        child_values = values[tree._child_pad[nodes]]          # (n, width, 2)
        values[nodes] = np.einsum("nw,nwp->np", probs[nodes], child_values)
    return float(values[tree.root, 0]), float(values[tree.root, 1])


def own_profit_best_response(tree: SequentialAuctionTree, player: int,
                             opponent_policy: Policy,
                             forced: dict[int, int] | None = None,
                             ) -> tuple[Policy, float]:
    """Exact own-profit best response for ``player`` against a fixed rival.

    ``forced`` pins chosen information sets to a given action; every other decision
    still best-responds. That yields the exact value of *committing* to a move (say,
    an opening jump bid) rather than the value of playing freely, which is what
    measures whether the commitment pays.

    Returns the (pure) best-response policy and the own profit it earns.
    """
    policies: list[Policy] = [None, None]  # type: ignore[list-item]
    policies[player] = tree.uniform_policy(player)  # unused: own reach is skipped
    policies[1 - player] = opponent_policy
    probs = tree._action_probs(policies[0], policies[1], skip_player=player)
    cf_reach = tree._reach(probs)

    override = np.full(tree.num_infosets(player), -1, dtype=np.int64)
    for index, action in (forced or {}).items():
        actions = tree.infoset_actions(player, index)
        if action not in actions:
            raise ValueError(
                f"action {action} is not legal at information set {index} "
                f"of player {player}; legal actions are {actions}")
        override[index] = actions.index(action)

    values = np.zeros(tree.num_nodes)
    values[tree._terminals] = tree._profits[tree._terminals, player]
    valid = tree._infoset_valid[player]
    chosen = np.zeros(tree.num_infosets(player), dtype=np.int64)

    for depth in range(tree.max_depth - 1, -1, -1):
        passive = tree._passive_by_depth[player][depth]
        if len(passive) > 0:
            child_values = values[tree._child_pad[passive]]
            values[passive] = np.einsum("nw,nw->n", probs[passive], child_values)

        own = tree._decision_by_depth[player][depth]
        if len(own) == 0:
            continue
        indices = tree._node_infoset[own]
        child_values = values[tree._child_pad[own]]            # (n, width)
        weighted = cf_reach[own, None] * child_values
        # Sum each history's contribution into its information set: one action is
        # chosen per set, not per history.
        action_values = np.zeros((tree.num_infosets(player), tree.width))
        np.add.at(action_values, indices, weighted)
        action_values[~valid] = -np.inf
        best = np.argmax(action_values, axis=1)
        # Only sets at this depth are read below, so a blanket override is safe.
        best = np.where(override >= 0, np.maximum(override, 0), best)
        chosen[tree._infosets_by_depth[player][depth]] = (
            best[tree._infosets_by_depth[player][depth]])
        values[own] = child_values[np.arange(len(own)), best[indices]]

    best_response = [np.zeros(len(actions))
                     for actions in tree._infoset_actions[player]]
    for index, action in enumerate(chosen):
        best_response[index][action] = 1.0
    return best_response, float(values[tree.root])


def opening_infosets(tree: SequentialAuctionTree, player: int) -> list[int]:
    """Information sets where ``player`` makes its first bidding decision.

    One per signal draw the bidder could be holding when it opens.
    """
    for depth in range(tree.max_depth + 1):
        indices = tree._infosets_by_depth[player][depth]
        if len(indices) > 0:
            return [int(index) for index in indices]
    return []


def opening_bid_profit_curve(tree: SequentialAuctionTree, opponent_policy: Policy,
                             player: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Own profit from committing to each possible opening bid, rival held fixed.

    For every level on the bid grid (plus the option to wait), the bidder is forced
    to open there and then plays an exact best response for the rest of the auction.
    The result is the preemption incentive curve: how much jump-bidding is actually
    worth. A peaked curve means preempting strictly pays and pins the opening down; a
    flat one means the bidder is indifferent across openings, so nothing pins it down
    and no solver should be expected to.

    Returns ``(bid amounts, profits)``. The bid amount for the "wait" option is NaN,
    since waiting names no price.
    """
    opening = opening_infosets(tree, player)
    actions = tree.infoset_actions(player, opening[0])
    if any(tree.infoset_actions(player, index) != actions for index in opening):
        raise ValueError("opening information sets do not share a legal action set")

    bids, profits = [], []
    for action in actions:
        _, value = own_profit_best_response(
            tree, player, opponent_policy,
            forced={index: action for index in opening})
        bids.append(np.nan if action == tree.pass_action
                    else float(tree.bid_amount[action]))
        profits.append(value)
    return np.array(bids), np.array(profits)


def own_profit_fictitious_play(
    tree: SequentialAuctionTree,
    max_iterations: int = FP_MAX_ITERATIONS,
    tolerance: float = FP_TOLERANCE,
    initial_policies: list[Policy] | None = None,
    callback=None,
) -> dict:
    """Solve the sequential auction's own-profit equilibrium by fictitious play.

    ``initial_policies`` restarts the search somewhere other than uniform. A
    general-sum game can carry several equilibria, so a comparative static is only
    trustworthy if it survives restarts; see the equilibrium-selection study in
    ``experiments/sequential_general_sum.py``.

    ``callback(iteration, policy0, policy1, nashconv)`` fires each round if given. It
    exists so a caller can watch an *economic* statistic settle: NashConv falling is
    necessary but not sufficient, and the quantity a paper reports may stabilize long
    before the equilibrium itself does (or, worse, may still be drifting once it does).
    """
    if initial_policies is None:
        avg = [tree.uniform_policy(0), tree.uniform_policy(1)]
    else:
        avg = [[probs.copy() for probs in initial_policies[0]],
               [probs.copy() for probs in initial_policies[1]]]
    nashconv_trace: list[tuple[int, float]] = []

    for iteration in range(1, max_iterations + 1):
        br0, br_value0 = own_profit_best_response(tree, 0, avg[1])
        br1, br_value1 = own_profit_best_response(tree, 1, avg[0])
        value0, value1 = policy_value(tree, avg[0], avg[1])
        nashconv = (br_value0 - value0) + (br_value1 - value1)
        nashconv_trace.append((iteration, nashconv))
        if callback is not None:
            callback(iteration, avg[0], avg[1], nashconv)
        if nashconv < tolerance:
            break
        weight = 1.0 / (iteration + 1)
        for player, best_response in ((0, br0), (1, br1)):
            for index, probs in enumerate(best_response):
                avg[player][index] *= 1.0 - weight
                avg[player][index] += weight * probs

    final_nashconv = nashconv_trace[-1][1]
    converged = final_nashconv < tolerance
    if not converged:
        warnings.warn(
            f"sequential fictitious play did not reach tolerance {tolerance:g} in "
            f"{nashconv_trace[-1][0]} iterations (NashConv={final_nashconv:.3e}); "
            "the returned values are a near-equilibrium, not converged.",
            RuntimeWarning,
            stacklevel=2,
        )
    value0, value1 = policy_value(tree, avg[0], avg[1])
    return {
        "policy0": avg[0],
        "policy1": avg[1],
        "value0": value0,
        "value1": value1,
        "nashconv": final_nashconv,
        "converged": converged,
        "iterations": nashconv_trace[-1][0],
        "nashconv_trace": nashconv_trace,
    }


def equilibrium_statistics(tree: SequentialAuctionTree, policy0: Policy,
                           policy1: Policy) -> dict:
    """Economic read-out of a solved sequential auction.

    ``p_rival_deterred`` is the headline: the chance that bidder 1, having watched
    bidder 0 open with a bid, concedes on the spot rather than contest it. That is
    the preemption channel the sealed auction cannot express, and it is what the
    empirical takeover literature argues a toehold is really buying.
    """
    probs = tree._action_probs(policy0, policy1)
    reach = tree._reach(probs)
    value0, value1 = policy_value(tree, policy0, policy1)

    terminals = tree._terminals
    terminal_reach = reach[terminals]
    won_by = np.array([-1 if tree.winner[t] is None else tree.winner[t]
                       for t in terminals])
    p_no_sale = float(terminal_reach[won_by == -1].sum())
    p_win = [float(terminal_reach[won_by == p].sum()) for p in range(2)]
    sold = won_by >= 0
    expected_price = float((terminal_reach[sold] * tree.price[terminals][sold]).sum())

    opening = tree._decision_by_depth[0][tree.first_bid_depth]
    open_probs = probs[opening]
    open_reach = reach[opening]
    is_pass = np.array([[action == tree.pass_action
                         for action in tree.infoset_actions(0, index)]
                        + [False] * (tree.width - len(tree.infoset_actions(0, index)))
                        for index in tree._node_infoset[opening]])
    # An opening information set offers every bid index in order, so column j is bid
    # index j. Its money value is bid_amount[j], which is only the index itself when
    # bid_step is 1.
    money = np.zeros(tree.width)
    money[:tree.num_bids] = tree.bid_amount
    p_open_wait = float((open_reach[:, None] * open_probs * is_pass).sum())
    raise_mass = open_reach[:, None] * open_probs * ~is_pass
    opening_bid_mass = float(raise_mass.sum())
    opening_bid = float((raise_mass * money[None, :]).sum())

    # Bidder 1's first turn, facing a live bid: does it fold on the spot?
    responding = tree._decision_by_depth[1][tree.first_bid_depth + 1]
    standing = np.array([tree._infoset_standing[1][index]
                         for index in tree._node_infoset[responding]])
    responding = responding[standing]
    p_opened = float(reach[responding].sum())
    fold = np.array([[action == tree.pass_action
                      for action in tree.infoset_actions(1, index)]
                     + [False] * (tree.width - len(tree.infoset_actions(1, index)))
                     for index in tree._node_infoset[responding]])
    p_deterred = float((reach[responding, None] * probs[responding] * fold).sum())

    return {
        "value0": value0,
        "value1": value1,
        "p_win0": p_win[0],
        "p_win1": p_win[1],
        "p_no_sale": p_no_sale,
        "expected_price": expected_price,
        "expected_price_given_sale": (
            expected_price / (p_win[0] + p_win[1]) if p_win[0] + p_win[1] > 0 else 0.0),
        "p_open_wait0": p_open_wait,
        "expected_opening_bid0": (
            opening_bid / opening_bid_mass if opening_bid_mass > 0 else 0.0),
        "p_rival_deterred": p_deterred / p_opened if p_opened > 0 else 0.0,
    }
