"""
Agent graders for SupplyChainEnv — easy / medium / hard.
Self-contained: includes all environment logic inline.
Scores are strictly in (0.01, 0.99).
"""

from __future__ import annotations
import argparse
import random
import math
import sys
import os


# ===========================================================================
# INLINE ENVIRONMENT — no external imports needed
# ===========================================================================

class OrderItem:
    def __init__(self, product_id: int, quantity: int, supplier_id: int):
        self.product_id = product_id
        self.quantity = quantity
        self.supplier_id = supplier_id


class SupplyChainAction:
    def __init__(self, orders=None):
        self.orders = orders or []


class Obs:
    def __init__(self, inventory, in_transit, demand_forecast,
                 supplier_prices, supplier_lead_times, stockouts,
                 step_cost, reward, done, metadata):
        self.inventory = inventory
        self.in_transit = in_transit
        self.demand_forecast = demand_forecast
        self.supplier_prices = supplier_prices
        self.supplier_lead_times = supplier_lead_times
        self.stockouts = stockouts
        self.step_cost = step_cost
        self.reward = reward
        self.done = done
        self.metadata = metadata


TASK_CONFIGS = {
    "easy":   {"num_warehouses": 1, "num_products": 1, "num_suppliers": 2, "max_steps": 20, "demand_noise": 0.05, "seasonal": False, "supplier_delay_prob": 0.0},
    "medium": {"num_warehouses": 1, "num_products": 5, "num_suppliers": 3, "max_steps": 40, "demand_noise": 0.15, "seasonal": True,  "supplier_delay_prob": 0.0},
    "hard":   {"num_warehouses": 3, "num_products": 10,"num_suppliers": 4, "max_steps": 60, "demand_noise": 0.25, "seasonal": True,  "supplier_delay_prob": 0.2},
}


class SimpleSupplyChainEnv:
    def __init__(self, task_id="easy", seed=42):
        self.task_id = task_id
        self.cfg = TASK_CONFIGS[task_id]
        self._rng = random.Random(seed)
        self._state = {}

    def reset(self, seed=42):
        self._rng = random.Random(seed)
        cfg = self.cfg
        NW = cfg["num_warehouses"]
        NP = cfg["num_products"]
        NS = cfg["num_suppliers"]
        max_lt = 3

        self.base_demand = [[self._rng.uniform(10, 50) for _ in range(NP)] for _ in range(NW)]
        self.supplier_prices = []
        self.supplier_lead_times = []
        for p in range(NP):
            prices = sorted([self._rng.uniform(1.0, 5.0) for _ in range(NS)], reverse=True)
            lead_times = sorted([self._rng.randint(1, max_lt) for _ in range(NS)])
            self.supplier_prices.append(prices)
            self.supplier_lead_times.append(lead_times)

        self.inventory = [[self.base_demand[w][p] * 2 for p in range(NP)] for w in range(NW)]
        self.in_transit = [[[0.0] * NP for _ in range(NW)] for _ in range(max_lt)]
        self.season_phase = 0.0
        self.step_count = 0
        self.NW, self.NP, self.NS = NW, NP, NS
        self.max_lt = max_lt

        return self._make_obs([[0.0]*NP for _ in range(NW)], 0.0, False, None)

    def step(self, action):
        cfg = self.cfg
        NW, NP, NS = self.NW, self.NP, self.NS

        # receive arrivals
        arrivals = self.in_transit[0]
        for w in range(NW):
            for p in range(NP):
                self.inventory[w][p] += arrivals[w][p]
        self.in_transit = self.in_transit[1:] + [[[0.0]*NP for _ in range(NW)]]

        # place orders
        order_cost = 0.0
        for order in action.orders:
            pid, sid, qty = order.product_id, order.supplier_id, order.quantity
            if pid >= NP or sid >= NS or qty <= 0:
                continue
            price = self.supplier_prices[pid][sid]
            lt = self.supplier_lead_times[pid][sid]
            if cfg["supplier_delay_prob"] > 0 and self._rng.random() < cfg["supplier_delay_prob"]:
                lt = min(lt + 1, self.max_lt - 1)
            lt = max(0, min(lt, self.max_lt - 1))
            per_wh = qty / NW
            for w in range(NW):
                self.in_transit[lt][w][pid] += per_wh
            order_cost += price * qty

        # demand
        self.season_phase += 2 * math.pi / 20
        season_factor = 1.0 + (0.3 * (0.5 + 0.5 * math.sin(self.season_phase)) if cfg["seasonal"] else 0.0)
        demand = []
        for w in range(NW):
            row = []
            for p in range(NP):
                base = self.base_demand[w][p] * season_factor
                noise = self._rng.gauss(0, base * cfg["demand_noise"])
                row.append(max(0.0, base + noise))
            demand.append(row)

        # fulfill
        stockouts = [[0.0]*NP for _ in range(NW)]
        fulfilled = 0.0
        for w in range(NW):
            for p in range(NP):
                d = demand[w][p]
                avail = self.inventory[w][p]
                self.inventory[w][p] = max(0.0, avail - d)
                stockouts[w][p] = max(0.0, d - avail)
                fulfilled += min(d, avail)

        total_demand = sum(demand[w][p] for w in range(NW) for p in range(NP))
        total_stockout = sum(stockouts[w][p] for w in range(NW) for p in range(NP))
        service_level = fulfilled / max(1.0, total_demand)
        stockout_severity = total_stockout / max(1.0, total_demand)
        total_inv = sum(self.inventory[w][p] for w in range(NW) for p in range(NP))
        expected_inv = total_demand * 1.5
        excess_ratio = max(0.0, total_inv - expected_inv) / max(1.0, expected_inv)
        holding_penalty = excess_ratio * 0.05
        reward = max(-1.0, min(1.0, service_level * 0.9 - stockout_severity * 0.5 - holding_penalty - 0.05))

        self.step_count += 1
        done = self.step_count >= cfg["max_steps"]
        step_cost = order_cost + total_inv * 0.01 + total_stockout * 3.0

        return self._make_obs(stockouts, step_cost, done, reward)

    def _make_obs(self, stockouts, step_cost, done, reward):
        cfg = self.cfg
        NW, NP = self.NW, self.NP
        season_factor = 1.0 + (0.3 * (0.5 + 0.5 * math.sin(self.season_phase)) if cfg["seasonal"] else 0.0)
        demand_forecast = []
        for w in range(NW):
            row = []
            for p in range(NP):
                base = self.base_demand[w][p] * season_factor
                noise = self._rng.gauss(0, base * cfg["demand_noise"] * 2)
                row.append(max(0.0, round(base + noise, 2)))
            demand_forecast.append(row)
        in_transit_next = [[round(self.in_transit[0][w][p], 2) for p in range(NP)] for w in range(NW)]
        return Obs(
            inventory=[[round(self.inventory[w][p], 2) for p in range(NP)] for w in range(NW)],
            in_transit=in_transit_next,
            demand_forecast=demand_forecast,
            supplier_prices=self.supplier_prices,
            supplier_lead_times=self.supplier_lead_times,
            stockouts=[[round(x, 2) for x in row] for row in stockouts],
            step_cost=round(step_cost, 4),
            done=done,
            reward=reward,
            metadata={"step": self.step_count, "task_id": self.task_id},
        )


# ===========================================================================
# GREEDY POLICY
# ===========================================================================

def greedy_restock_policy(obs, safety_factor=1.5):
    orders = []
    NW = len(obs.inventory)
    NP = len(obs.inventory[0]) if NW > 0 else 0
    NS = len(obs.supplier_prices[0]) if NP > 0 else 0
    for p in range(NP):
        total_inv = sum(obs.inventory[w][p] for w in range(NW))
        total_transit = sum(obs.in_transit[w][p] for w in range(NW))
        total_forecast = sum(obs.demand_forecast[w][p] for w in range(NW))
        gap = max(0.0, total_forecast * safety_factor - total_inv - total_transit)
        if gap > 1.0:
            orders.append(OrderItem(product_id=p, quantity=int(round(gap)), supplier_id=NS - 1))
    return SupplyChainAction(orders=orders)


# ===========================================================================
# GRADERS
# ===========================================================================

def run_grader(task_id: str, seed: int = 42, verbose: bool = False) -> float:
    env = SimpleSupplyChainEnv(task_id=task_id, seed=seed)
    cfg = TASK_CONFIGS[task_id]
    obs = env.reset(seed=seed)

    total_stockout_events = 0
    total_steps = 0
    episode_reward = 0.0

    while not obs.done and total_steps < cfg["max_steps"]:
        action = greedy_restock_policy(obs)
        obs = env.step(action)
        total_steps += 1
        if any(obs.stockouts[w][p] > 0 for w in range(len(obs.stockouts)) for p in range(len(obs.stockouts[0]))):
            total_stockout_events += 1
        if obs.reward is not None:
            episode_reward += obs.reward

    max_possible = cfg["max_steps"]
    raw_score = (episode_reward + max_possible) / (2 * max_possible)
    stockout_rate = total_stockout_events / max(1, total_steps)
    final_score = raw_score - stockout_rate * 0.3

    # STRICTLY between 0 and 1
    final_score = max(0.01, min(0.99, float(final_score)))

    if verbose:
        print(f"  Task: {task_id} | Steps: {total_steps} | Score: {final_score:.4f}")

    return round(final_score, 4)


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
