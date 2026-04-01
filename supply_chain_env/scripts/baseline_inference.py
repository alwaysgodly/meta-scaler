"""
Baseline inference script for SupplyChainEnv.

Runs the greedy restocking agent on all 3 tasks and reports reproducible scores.

Usage:
    python scripts/baseline_inference.py
    python scripts/baseline_inference.py --task hard --seed 123
"""

from __future__ import annotations
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tasks"))

from graders import GRADERS, grade_easy, grade_medium, grade_hard
from supply_chain_env.env import SupplyChainEnv, SupplyChainAction, OrderItem, TASK_CONFIGS


def run_full_episode_display(task_id: str, seed: int = 42):
    """Run and display a full episode step-by-step."""
    from graders import greedy_restock_policy

    env = SupplyChainEnv(task_id=task_id, seed=seed)
    cfg = TASK_CONFIGS[task_id]

    print(f"\n{'='*60}")
    print(f"Task: {task_id.upper()} — {cfg['description']}")
    print(f"Warehouses: {cfg['num_warehouses']} | Products: {cfg['num_products']} | "
          f"Suppliers: {cfg['num_suppliers']} | Steps: {cfg['max_steps']}")
    print(f"{'='*60}")

    obs = env.reset(seed=seed)
    print(f"\nInitial inventory: {obs.inventory}")
    print(f"Supplier prices:   {obs.supplier_prices}")
    print(f"Lead times:        {obs.supplier_lead_times}")
    print()

    step = 0
    total_reward = 0.0

    while not obs.done:
        action = greedy_restock_policy(obs, cfg)
        obs = env.step(action)
        step += 1
        if obs.reward:
            total_reward += obs.reward

        orders_summary = [(o.product_id, o.quantity, o.supplier_id) for o in action.orders]

        if step <= 5 or step % 10 == 0 or obs.done:
            print(f"Step {step:3d} | reward={obs.reward:+.3f} | cost={obs.step_cost:.2f} | "
                  f"orders={orders_summary} | stockouts={obs.stockouts}")

    print(f"\nTotal reward: {total_reward:.3f}")
    return total_reward


def main():
    parser = argparse.ArgumentParser(description="SupplyChainEnv baseline inference")
    parser.add_argument("--task", choices=["easy", "medium", "hard", "all"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--display", action="store_true", help="Show step-by-step output")
    args = parser.parse_args()

    tasks = ["easy", "medium", "hard"] if args.task == "all" else [args.task]

    if args.display:
        for t in tasks:
            run_full_episode_display(t, seed=args.seed)

    print(f"\n{'='*60}")
    print("REPRODUCIBLE BASELINE SCORES (seed={})".format(args.seed))
    print(f"{'='*60}")

    scores = {}
    for t in tasks:
        score = GRADERS[t](seed=args.seed, verbose=False)
        scores[t] = score

    for t, s in scores.items():
        bar = "█" * int(s * 30) + "░" * (30 - int(s * 30))
        print(f"  {t:8s}  [{bar}]  {s:.4f}")

    print(f"\nBaseline agent: Greedy Restock (safety_factor=1.5, cheapest supplier)")
    print("Scores are deterministic given the same seed.")


if __name__ == "__main__":
    main()
