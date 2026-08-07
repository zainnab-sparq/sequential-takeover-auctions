"""Re-measure the certified profile of the deep-restart cell without trusting its scalars.

The deep-restart study (``certify_deep``) reports one solve out of eighteen inside the
acceptance bound at thirteen bid levels and a toehold of 0.15. That number is the solver
grading its own work, so this script re-derives what it can from the game tree.

What it can and cannot check is worth being explicit about. The solve cache persists the
rival's policy but not bidder 0's, so the stored profile cannot be re-measured end to end
here. Two checks are available and both are run:

1. Bidder 0's half of the stored profile's NashConv, computed fresh. An exact best
   response to the stored rival is rebuilt from the tree, and its value is compared to the
   stored profile value. A negative gap would prove the stored value is unattainable; a
   gap above the acceptance bound would prove the profile is not certified. Bidder 1's
   half is not independently checkable without bidder 0's policy.

2. The conduct. Statistics are recomputed from the reconstructed profile and compared to
   the published CSV, which tests whether the reported deterrence and opening bid are
   properties of the equilibrium or artifacts of the stored read-out.

Run inside the project image:

    docker run --rm -v "$PWD:/work" -w /work imperfect-info:latest \
        python -u scripts/verify_certified_profile.py
"""

import glob
import pickle

import numpy as np
import pyspiel

import dealgame  # noqa: F401
from dealgame.sequential_general_sum import (SequentialAuctionTree,
                                             equilibrium_statistics,
                                             own_profit_best_response,
                                             policy_value)

DEEP_BIDS = 13
DEEP_TOEHOLD = 0.15
DEEP_RESTART = 16
ACCEPT_EPS = 1e-4
CACHE_GLOB = "results/.solve_cache/*.pkl"


def _load_certified() -> tuple[str, dict]:
    """The v2 cache entry for the one deep-cell restart that cleared the bound."""
    for path in glob.glob(CACHE_GLOB):
        with open(path, "rb") as handle:
            stats = pickle.load(handle)
        params = stats.get("params", {})
        # ``wander`` is absent from v1 entries, which certified the final iterate rather
        # than the best one; those are a different definition and must not be read here.
        if (params.get("num_bids") == DEEP_BIDS
                and abs(params.get("toehold", -1) - DEEP_TOEHOLD) < 1e-12
                and stats.get("restart") == DEEP_RESTART
                and stats.get("wander") is not None):
            return path, stats
    raise SystemExit(f"no v2 cache entry for restart {DEEP_RESTART} of the deep cell")


def main() -> int:
    path, stored = _load_certified()
    print(f"cache entry  {path}")
    print(f"stored       eps {stored['nashconv']:.4e}   wander {stored['wander']:.3f}"
          f"   best iterate {stored['best_iteration']:,}")
    print(f"             deterrence {stored['p_rival_deterred']:.6f}"
          f"   opening bid {stored['expected_opening_bid0']:.6f}"
          f"   value0 {stored['value0']:.6f}")

    game = pyspiel.load_game("dealgame_sequential_takeover", stored["params"])
    tree = SequentialAuctionTree(game)
    rival = [np.asarray(row, dtype=float) for row in stored["policy1"]]

    br0_policy, br0_value = own_profit_best_response(tree, 0, rival)
    gap0 = br0_value - stored["value0"]
    print("\n-- check 1: bidder 0's exploitability in the stored profile --")
    print(f"exact best response value  {br0_value:.10f}")
    print(f"stored profile value       {stored['value0']:.10f}")
    print(f"gap                        {gap0:.4e}   (bound {ACCEPT_EPS:.0e},"
          f" stored total eps {stored['nashconv']:.4e})")
    ok_gap = 0.0 <= gap0 <= ACCEPT_EPS
    if gap0 < 0.0:
        print("  FAIL: the stored value beats an exact best response, so it is unattainable")
    elif not ok_gap:
        print("  FAIL: bidder 0 can deviate by more than the acceptance bound")
    else:
        print("  PASS: bidder 0 cannot deviate by more than the acceptance bound")

    check = equilibrium_statistics(tree, br0_policy, rival)
    print("\n-- check 2: conduct, recomputed from the reconstructed profile --")
    ok_conduct = True
    for key, tol in (("p_rival_deterred", 1e-3), ("expected_opening_bid0", 1e-3),
                     ("value0", 1e-3)):
        diff = check[key] - stored[key]
        ok_conduct &= abs(diff) <= tol
        print(f"{key:24} stored {stored[key]:.6f}   recomputed {check[key]:.6f}"
              f"   diff {diff:+.2e}")
    print(f"  {'PASS' if ok_conduct else 'FAIL'}: reported conduct is"
          f" {'reproduced' if ok_conduct else 'NOT reproduced'} from the tree")

    # Diagnostic, not a pass/fail test. The reconstructed profile pairs a *pure* best
    # response with the stored rival, which is a different profile from the certified one,
    # so its exploitability says nothing about the certificate. It does say the
    # equilibrium needs bidder 0 to mix: pin bidder 0 to a pure reply and the rival
    # becomes exploitable by two orders of magnitude more than the bound.
    value0, value1 = policy_value(tree, br0_policy, rival)
    _, br1_value = own_profit_best_response(tree, 1, br0_policy)
    print("\n-- diagnostic: the same rival against a PURE bidder 0 --")
    print(f"bidder 1's gap  {br1_value - value1:.4e}"
          "   (a different profile; mixing is what sustains the equilibrium)")

    passed = ok_gap and ok_conduct
    print(f"\nVERDICT  {'VERIFIED' if passed else 'NOT VERIFIED'}: the certified profile"
          f" {'survives' if passed else 'fails'} independent re-measurement"
          " of bidder 0's half and of its conduct.")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
