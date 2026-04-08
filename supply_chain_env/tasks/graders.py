"""
Agent graders for SupplyChainEnv — easy / medium / hard.
Scores are strictly in (0.0, 1.0).

Usage:
    python tasks/graders.py --task all --seed 42
"""

from __future__ import annotations
import argparse
import sys
import os

# Works both locally and in Docker
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in [
    os.path.join(BASE_DIR, "src"),
    os.path.join(BASE_DIR, "tasks"),
    "/app/src",
    "/app/tasks",
]:
    if p not in sys.path:
        sys.path.insert(0, p)

from supply_chain_env.env import SupplyChainEnv, SupplyChainAction, OrderItem, TASK_CONFIGS


def greedy_restock_policy(obs, cfg: dict, safety_factor: float = 1.5) -> SupplyChainAction:
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
            best_supplier = NS - 1
            qty = int(round(gap))
            orders.append(OrderItem(product_id=p, quantity=qty, supplier_id=best_supplier))

    return SupplyChainAction(orders=orders)


def run_grader(task_id: str, seed: int = 42, verbose: bool = False) -> float:
    env = SupplyChainEnv(task_id=task_id, seed=seed)
    cfg = TASK_CONFIGS[task_id]

    obs = env.reset(seed=seed)
    total_stockout_events = 0
    total_steps = 0
    episode_reward = 0.0

    while not obs.done:
        action = greedy_restock_policy(obs, cfg)
        obs = env.step(action)
        total_steps += 1

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
            print(f"  Step {step:3d} | reward={obs.reward:.3f} | stockout_events={total_stockout_events}")

    max_possible = cfg["max_steps"]
    raw_score = (episode_reward + max_possible) / (2 * max_possible)
    raw_score = max(0.0, min(1.0, raw_score))

    stockout_rate = total_stockout_events / max(1, total_steps)
    stockout_penalty = stockout_rate * 0.3

    final_score = raw_score - stockout_penalty

    # Ensure strictly between 0 and 1 (never exactly 0.0 or 1.0)
    final_score = max(0.001, min(0.999, final_score))

    if verbose:
        print(f"\n  Task: {task_id}")
        print(f"  Episode reward: {episode_reward:.3f}")
        print(f"  Stockout events: {total_stockout_events}/{total_steps} ({stockout_rate:.1%})")
        print(f"  Final score: {final_score:.4f}")

    return round(final_score, 4)


def grade_easy(seed: int = 42, verbose: bool = False) -> float:
    return run_grader("easy", seed=seed, verbose=verbose)


def grade_medium(seed: int = 42, verbose: bool = False) -> float:
    return run_grader("medium", seed=seed, verbose=verbose)


def grade_hard(seed: int = 42, verbose: bool = False) -> float:
    return run_grader("hard", seed=seed, verbose=verbose)


GRADERS = {
    "easy": grade_easy,
    "medium": grade_medium,
    "hard": grade_hard,
}

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
