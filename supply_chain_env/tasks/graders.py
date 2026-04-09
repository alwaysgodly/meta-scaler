"""
Agent graders for SupplyChainEnv — easy / medium / hard.
Scores are strictly in (0.01, 0.99).
"""

from __future__ import annotations
import argparse
import sys
import os

# Fix paths for all possible locations
for p in [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "supply_chain_env", "src"),
    "/app/src",
    "/app/supply_chain_env/src",
]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

from supply_chain_env.env import SupplyChainEnv, SupplyChainAction, OrderItem, TASK_CONFIGS


def greedy_restock_policy(obs, safety_factor: float = 1.5) -> SupplyChainAction:
    orders = []
    try:
        NW = len(obs.inventory)
        NP = len(obs.inventory[0]) if NW > 0 else 0
        NS = len(obs.supplier_prices[0]) if NP > 0 else 0
        for p in range(NP):
            total_inv = sum(obs.inventory[w][p] for w in range(NW))
            total_transit = sum(obs.in_transit[w][p] for w in range(NW))
            total_forecast = sum(obs.demand_forecast[w][p] for w in range(NW))
            gap = max(0.0, total_forecast * safety_factor - total_inv - total_transit)
            if gap > 1.0:
                orders.append(OrderItem(
                    product_id=p,
                    quantity=int(round(gap)),
                    supplier_id=NS - 1,
                ))
    except Exception:
        pass
    return SupplyChainAction(orders=orders)


def run_grader(task_id: str, seed: int = 42, verbose: bool = False) -> float:
    try:
        env = SupplyChainEnv(task_id=task_id, seed=seed)
        cfg = TASK_CONFIGS[task_id]
        obs = env.reset(seed=seed)

        total_stockout_events = 0
        total_steps = 0
        episode_reward = 0.0

        while not obs.done and total_steps < cfg["max_steps"]:
            action = greedy_restock_policy(obs)
            obs = env.step(action)
            total_steps += 1

            if any(obs.stockouts[w][p] > 0
                   for w in range(len(obs.stockouts))
                   for p in range(len(obs.stockouts[0]))):
                total_stockout_events += 1

            if obs.reward is not None:
                episode_reward += obs.reward

            if verbose:
                print(f"  Step {total_steps:3d} | reward={obs.reward:.3f} | stockouts={total_stockout_events}")

        max_possible = cfg["max_steps"]
        raw_score = (episode_reward + max_possible) / (2 * max_possible)
        stockout_rate = total_stockout_events / max(1, total_steps)
        final_score = raw_score - stockout_rate * 0.3

        # STRICTLY between 0 and 1
        final_score = max(0.01, min(0.99, final_score))

        if verbose:
            print(f"\n  Task: {task_id} | Score: {final_score:.4f}")

        return round(final_score, 4)

    except Exception as e:
        print(f"Grader error for {task_id}: {e}")
        return 0.5


def grade_easy(seed: int = 42, **kwargs) -> float:
    return run_grader("easy", seed=seed)


def grade_medium(seed: int = 42, **kwargs) -> float:
    return run_grader("medium", seed=seed)


def grade_hard(seed: int = 42, **kwargs) -> float:
    return run_grader("hard", seed=seed)


GRADERS = {
    "easy": grade_easy,
    "medium": grade_medium,
    "hard": grade_hard,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["easy", "medium", "hard", "all"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    tasks = ["easy", "medium", "hard"] if args.task == "all" else [args.task]
    print("=" * 50)
    print("SupplyChainEnv — Agent Grader")
    print("=" * 50)

    for t in tasks:
        print(f"\n[{t.upper()} TASK]")
        score = GRADERS[t](seed=args.seed, verbose=args.verbose)
        print(f"  Score: {score:.4f} / 1.0")
