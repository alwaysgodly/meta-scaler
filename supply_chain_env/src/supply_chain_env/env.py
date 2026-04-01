"""
SupplyChainEnv — A real-world supply chain restocking OpenEnv environment.

An AI agent manages inventory across warehouses, deciding what to restock,
how much, and from which supplier, balancing cost vs stockout risk.
"""

from __future__ import annotations

import random
import uuid
from typing import Any, Dict, List, Optional

from openenv.core import Action, Environment, Observation, State
from pydantic import Field


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class OrderItem(State):
    """A single order line: product + quantity from a supplier."""
    product_id: int = Field(ge=0, description="Product index to restock")
    quantity: int = Field(ge=0, le=500, description="Units to order (0 = no order)")
    supplier_id: int = Field(ge=0, description="Supplier index to order from")


class SupplyChainAction(Action):
    """Agent's action: a list of order items across products and suppliers."""
    orders: List[OrderItem] = Field(
        default_factory=list,
        description="List of restocking orders to place this step",
    )


class WarehouseObservation(Observation):
    """What the agent sees after each step."""
    # Inventory: shape [num_warehouses x num_products]
    inventory: List[List[float]] = Field(description="Current stock levels per warehouse/product")
    # In-transit orders arriving next step
    in_transit: List[List[float]] = Field(description="Units currently in transit per warehouse/product")
    # Demand forecast (noisy) for next step
    demand_forecast: List[List[float]] = Field(description="Noisy demand forecast per warehouse/product")
    # Supplier prices per product
    supplier_prices: List[List[float]] = Field(description="Price per unit per supplier per product")
    # Supplier lead times in steps
    supplier_lead_times: List[List[int]] = Field(description="Lead time (steps) per supplier per product")
    # Stockouts this step
    stockouts: List[List[float]] = Field(description="Unmet demand this step per warehouse/product")
    # Cost incurred this step
    step_cost: float = Field(description="Total cost incurred this step")
    # Info message
    message: str = Field(default="", description="Human-readable step info")


class SupplyChainState(State):
    """Internal environment state."""
    task_id: str = Field(default="easy")
    num_warehouses: int = Field(default=1)
    num_products: int = Field(default=1)
    num_suppliers: int = Field(default=2)
    max_steps: int = Field(default=30)
    inventory: List[List[float]] = Field(default_factory=list)
    in_transit: List[List[List[float]]] = Field(default_factory=list)  # [lead_time x wh x prod]
    supplier_prices: List[List[float]] = Field(default_factory=list)
    supplier_lead_times: List[List[int]] = Field(default_factory=list)
    base_demand: List[List[float]] = Field(default_factory=list)
    season_phase: float = Field(default=0.0)
    total_reward: float = Field(default=0.0)
    rng_seed: int = Field(default=42)


# ---------------------------------------------------------------------------
# Task configs
# ---------------------------------------------------------------------------

TASK_CONFIGS = {
    "easy": {
        "num_warehouses": 1,
        "num_products": 1,
        "num_suppliers": 2,
        "max_steps": 20,
        "demand_noise": 0.05,
        "seasonal": False,
        "supplier_delay_prob": 0.0,
        "description": "Single warehouse, 1 product, stable demand. Learn basic restocking.",
    },
    "medium": {
        "num_warehouses": 1,
        "num_products": 5,
        "num_suppliers": 3,
        "max_steps": 40,
        "demand_noise": 0.15,
        "seasonal": True,
        "supplier_delay_prob": 0.0,
        "description": "Single warehouse, 5 products, seasonal demand. Manage multiple SKUs.",
    },
    "hard": {
        "num_warehouses": 3,
        "num_products": 10,
        "num_suppliers": 4,
        "max_steps": 60,
        "demand_noise": 0.25,
        "seasonal": True,
        "supplier_delay_prob": 0.2,
        "description": "3 warehouses, 10 products, noisy seasonal demand + random supplier delays.",
    },
}


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class SupplyChainEnv(Environment[SupplyChainAction, WarehouseObservation, SupplyChainState]):
    """
    Supply Chain Restocking Environment for RL agents.

    The agent manages inventory across warehouses by placing orders
    each step. Reward = fulfilled demand value - ordering cost - stockout penalty.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, task_id: str = "easy", seed: Optional[int] = None):
        super().__init__()
        assert task_id in TASK_CONFIGS, f"task_id must be one of {list(TASK_CONFIGS.keys())}"
        self.task_id = task_id
        self.cfg = TASK_CONFIGS[task_id]
        self._seed = seed if seed is not None else 42
        self._state: Optional[SupplyChainState] = None
        self._rng = random.Random(self._seed)

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs) -> WarehouseObservation:
        if seed is not None:
            self._seed = seed
        self._rng = random.Random(self._seed)

        cfg = self.cfg
        NW = cfg["num_warehouses"]
        NP = cfg["num_products"]
        NS = cfg["num_suppliers"]
        max_lt = 3  # max lead time steps

        # Random base demand 10–50 per product per warehouse
        base_demand = [[self._rng.uniform(10, 50) for _ in range(NP)] for _ in range(NW)]

        # Supplier prices: cheaper supplier has longer lead time
        supplier_prices = []
        supplier_lead_times = []
        for p in range(NP):
            prices = sorted([self._rng.uniform(1.0, 5.0) for _ in range(NS)], reverse=True)
            lead_times = sorted([self._rng.randint(1, max_lt) for _ in range(NS)])
            supplier_prices.append(prices)
            supplier_lead_times.append(lead_times)

        # Initial inventory: 2 cycles worth
        inventory = [[base_demand[w][p] * 2 for p in range(NP)] for w in range(NW)]

        # Empty in-transit queue: shape [max_lt steps x NW x NP]
        in_transit = [[[0.0] * NP for _ in range(NW)] for _ in range(max_lt)]

        self._state = SupplyChainState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            task_id=self.task_id,
            num_warehouses=NW,
            num_products=NP,
            num_suppliers=NS,
            max_steps=cfg["max_steps"],
            inventory=inventory,
            in_transit=in_transit,
            supplier_prices=supplier_prices,
            supplier_lead_times=supplier_lead_times,
            base_demand=base_demand,
            season_phase=0.0,
            total_reward=0.0,
            rng_seed=self._seed,
        )

        return self._make_obs(stockouts=[[0.0] * NP for _ in range(NW)], step_cost=0.0, done=False)

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------
    def step(self, action: SupplyChainAction, timeout_s: Optional[float] = None, **kwargs) -> WarehouseObservation:
        s = self._state
        assert s is not None, "Call reset() before step()"

        cfg = self.cfg
        NW, NP, NS = s.num_warehouses, s.num_products, s.num_suppliers
        max_lt = len(s.in_transit)

        # 1. Receive arrivals (lead_time=1 slot arrives this step)
        arrivals = s.in_transit[0]
        for w in range(NW):
            for p in range(NP):
                s.inventory[w][p] += arrivals[w][p]

        # Shift transit queue forward
        s.in_transit = s.in_transit[1:] + [[[0.0] * NP for _ in range(NW)]]

        # 2. Place orders from action
        order_cost = 0.0
        for order in action.orders:
            pid = order.product_id
            sid = order.supplier_id
            qty = order.quantity

            if pid >= NP or sid >= NS or qty <= 0:
                continue

            price = s.supplier_prices[pid][sid]
            base_lt = s.supplier_lead_times[pid][sid]

            # Random delay on hard task
            if cfg["supplier_delay_prob"] > 0 and self._rng.random() < cfg["supplier_delay_prob"]:
                lt = min(base_lt + 1, max_lt - 1)
            else:
                lt = base_lt

            lt = max(0, min(lt, max_lt - 1))

            # Distribute evenly to all warehouses (simplified)
            per_wh = qty / NW
            for w in range(NW):
                s.in_transit[lt][w][pid] += per_wh

            order_cost += price * qty

        # 3. Compute actual demand with noise + seasonality
        s.season_phase += 2 * 3.14159 / 20  # one season cycle per 20 steps
        season_factor = 1.0 + (0.3 * (0.5 + 0.5 * _sin(s.season_phase)) if cfg["seasonal"] else 0.0)

        demand = []
        for w in range(NW):
            row = []
            for p in range(NP):
                base = s.base_demand[w][p] * season_factor
                noise = self._rng.gauss(0, base * cfg["demand_noise"])
                d = max(0.0, base + noise)
                row.append(d)
            demand.append(row)

        # 4. Fulfill demand, track stockouts
        stockouts = [[0.0] * NP for _ in range(NW)]
        fulfilled = 0.0
        for w in range(NW):
            for p in range(NP):
                d = demand[w][p]
                avail = s.inventory[w][p]
                sold = min(d, avail)
                s.inventory[w][p] = max(0.0, avail - d)
                stockouts[w][p] = max(0.0, d - avail)
                fulfilled += sold

        # 5. Holding cost: small penalty for excess inventory
        total_inventory = sum(s.inventory[w][p] for w in range(NW) for p in range(NP))
        holding_cost = total_inventory * 0.01

        # 6. Stockout penalty
        total_stockout = sum(stockouts[w][p] for w in range(NW) for p in range(NP))
        stockout_penalty = total_stockout * 3.0  # penalty > price to incentivize availability

        step_cost = order_cost + holding_cost + stockout_penalty

        # 7. Reward: service-level based with partial progress signals
        # Primary signal: fraction of demand fulfilled (0 to 1)
        total_demand = sum(demand[w][p] for w in range(NW) for p in range(NP))
        service_level = fulfilled / max(1.0, total_demand)
        # Stockout severity: fraction of demand that was unmet
        stockout_severity = total_stockout / max(1.0, total_demand)
        # Holding penalty: mild penalty for excessive inventory
        total_inv = sum(s.inventory[w][p] for w in range(NW) for p in range(NP))
        expected_inv = total_demand * 1.5  # healthy buffer
        excess_ratio = max(0.0, total_inv - expected_inv) / max(1.0, expected_inv)
        holding_penalty = excess_ratio * 0.05
        # Reward: [-1, 1] with partial progress
        # Good agent: service_level ~1, stockout ~0 -> reward ~0.85
        # Poor agent: service_level ~0.5, stockout ~0.5 -> reward ~0.1
        reward = float(max(-1.0, min(1.0,
            service_level * 0.9 - stockout_severity * 0.5 - holding_penalty - 0.05
        )))

        s.total_reward += reward
        s.step_count += 1
        done = s.step_count >= s.max_steps

        obs = self._make_obs(stockouts=stockouts, step_cost=step_cost, done=done, reward=reward)
        return obs

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------
    def state(self) -> SupplyChainState:
        assert self._state is not None, "Call reset() first"
        return self._state

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _make_obs(self, stockouts, step_cost, done, reward=None) -> WarehouseObservation:
        s = self._state
        NW, NP = s.num_warehouses, s.num_products
        cfg = self.cfg

        # Noisy demand forecast
        s.season_phase_preview = getattr(s, "season_phase_preview", s.season_phase)
        season_factor = 1.0 + (0.3 * (0.5 + 0.5 * _sin(s.season_phase)) if cfg["seasonal"] else 0.0)
        demand_forecast = []
        for w in range(NW):
            row = []
            for p in range(NP):
                base = s.base_demand[w][p] * season_factor
                noise = self._rng.gauss(0, base * cfg["demand_noise"] * 2)
                row.append(max(0.0, round(base + noise, 2)))
            demand_forecast.append(row)

        in_transit_next = [[round(s.in_transit[0][w][p], 2) for p in range(NP)] for w in range(NW)]

        return WarehouseObservation(
            inventory=[[round(s.inventory[w][p], 2) for p in range(NP)] for w in range(NW)],
            in_transit=in_transit_next,
            demand_forecast=demand_forecast,
            supplier_prices=s.supplier_prices,
            supplier_lead_times=s.supplier_lead_times,
            stockouts=[[round(x, 2) for x in row] for row in stockouts],
            step_cost=round(step_cost, 4),
            done=done,
            reward=reward,
            metadata={
                "step": s.step_count,
                "total_reward": round(s.total_reward, 4),
                "task_id": s.task_id,
            },
        )


def _sin(x: float) -> float:
    import math
    return math.sin(x)
