"""
Agent graders for SupplyChainEnv tasks.

Each grader runs the environment with a specific policy and returns a score 0.0–1.0.
- easy grader:   score based on zero stockouts over 20 steps
- medium grader: score based on managing 5 products, seasonal demand
- hard grader:   score based on multi-warehouse, noisy demand with delays

Usage:
    python graders.py --task easy
"""

from __future__ import annotations
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from supply_chain_env.env import SupplyChainEnv, SupplyChainAction, OrderItem


# ---------------------------------------------------------------------------
# Baseline policies
# ---------------------------------------------------------------------------

def greedy_restock_policy(obs, task_cfg: dict, safety_factor: float = 1.5) -> SupplyChainAction:
    """
    Simple greedy policy: order enough to cover forecast demand * safety_factor
    minus current inventory and in-transit, from cheapest supplier.
    """
    orders = []
    NW = len(obs.inventory)
    NP = len(obs.inventory[0]) if NW > 0 else 0
    NS = len(obs.supplier_prices[0]) if NP > 0 else 0

    for p in range(NP):
        total_inv = sum(obs.inventory[w][p] for w in range(NW))
        total_transit = sum(obs.in_transit[w][p] for w in range(NW))
        total_forecast = sum(obs.demand_forecast[w][p] for w in range(NW))

        target = total_forecast * safety_factor
        gap = max(0.0, target - total_inv - total_transit)

        if gap > 1.0:
            # Pick cheapest supplier (last index = cheapest by construction)
            best_supplier = NS - 1
            qty = int(round(gap))
            orders.append(OrderItem(product_id=p, quantity=qty, supplier_id=best_supplier))

    return SupplyChainAction(orders=orders)


# ---------------------------------------------------------------------------
# Graders
# ---------------------------------------------------------------------------

def run_grader(task_id: str, seed: int = 42, verbose: bool = False) -> float:
    """
    Run one episode with the greedy policy and compute a score 0.0–1.0.

    Scoring:
      - Base score from total_reward accumulated (normalized)
      - Penalties for repeated stockouts
    """
    env = SupplyChainEnv(task_id=task_id, seed=seed)
    from supply_chain_env.env import TASK_CONFIGS
    cfg = TASK_CONFIGS[task_id]

    obs = env.reset(seed=seed)
    total_stockout_events = 0
    total_steps = 0
    episode_reward = 0.0

    while not obs.done:
        action = greedy_restock_policy(obs, cfg)
        obs = env.step(action)
        total_steps += 1

        # Count stockout events (any warehouse/product with stockout > 0)
        stockout_event = any(
            obs.stockouts[w][p] > 0
            for w in range(len(obs.stockouts))
            for p in range(len(obs.stockouts[0]))
        )
        if stockout_event:
            total_stockout_events += 1

        if obs.reward is not None:
            episode_reward += obs.reward

        if verbose:
            step = obs.metadata.get("step", total_steps)
            print(f"  Step {step:3d} | reward={obs.reward:.3f} | cost={obs.step_cost:.2f} | stockout_events={total_stockout_events}")

    # Normalize reward to 0–1
    # Max possible: 1.0 per step * max_steps
    max_possible = cfg["max_steps"]
    # Shift from [-max, max] to [0, 1]
    raw_score = (episode_reward + max_possible) / (2 * max_possible)
    raw_score = max(0.0, min(1.0, raw_score))

    # Penalize stockout rate
    stockout_rate = total_stockout_events / max(1, total_steps)
    stockout_penalty = stockout_rate * 0.3

    final_score = max(0.0, min(1.0, raw_score - stockout_penalty))

    if verbose:
        print(f"\n  Task: {task_id}")
        print(f"  Episode reward: {episode_reward:.3f}")
        print(f"  Stockout events: {total_stockout_events}/{total_steps} ({stockout_rate:.1%})")
        print(f"  Raw score: {raw_score:.3f}")
        print(f"  Final score: {final_score:.3f}")

    return round(final_score, 4)


def grade_easy(seed: int = 42, verbose: bool = False) -> float:
    """Easy grader: single warehouse, 1 product, stable demand. Target: ≥ 0.7"""
    return run_grader("easy", seed=seed, verbose=verbose)


def grade_medium(seed: int = 42, verbose: bool = False) -> float:
    """Medium grader: 5 products, seasonal demand. Target: ≥ 0.5"""
    return run_grader("medium", seed=seed, verbose=verbose)


def grade_hard(seed: int = 42, verbose: bool = False) -> float:
    """Hard grader: 3 warehouses, 10 products, delays. Target: ≥ 0.35"""
    return run_grader("hard", seed=seed, verbose=verbose)


GRADERS = {
    "easy": grade_easy,
    "medium": grade_medium,
    "hard": grade_hard,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SupplyChainEnv graders")
    parser.add_argument("--task", choices=["easy", "medium", "hard", "all"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    tasks = ["easy", "medium", "hard"] if args.task == "all" else [args.task]

    print("=" * 50)
    print("SupplyChainEnv — Agent Grader")
    print("=" * 50)

    results = {}
    for t in tasks:
        print(f"\n[{t.upper()} TASK]")
        score = GRADERS[t](seed=args.seed, verbose=args.verbose)
        results[t] = score
        print(f"  Score: {score:.4f} / 1.0")

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for t, s in results.items():
        bar = "█" * int(s * 20) + "░" * (20 - int(s * 20))
        print(f"  {t:8s}  [{bar}]  {s:.4f}")
