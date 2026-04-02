# 📦 SupplyChainEnv

> A real-world **supply chain restocking** environment for RL agents, built on the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) framework.

The agent manages inventory across warehouses — deciding **what to order**, **how much**, and **from which supplier** — balancing cost efficiency against stockout risk and supplier lead times.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green)](https://python.org)
[![HuggingFace](https://img.shields.io/badge/🤗-Live%20on%20Spaces-yellow)](https://huggingface.co/spaces/alwaysgodly/supplychain)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## 🧠 Environment Overview

| Property | Value |
|---|---|
| **Domain** | Supply Chain / Logistics |
| **API** | OpenEnv (`step()` / `reset()` / `state()`) |
| **Action Space** | Structured — list of `OrderItem(product_id, quantity, supplier_id)` |
| **Observation Space** | Structured — inventory, demand forecast, prices, lead times, stockouts |
| **Reward Range** | `[-1.0, 1.0]` (partial progress signal) |
| **Tasks** | 3 (easy → medium → hard) |

---

## 🎯 Tasks

### 🟢 Easy
- **1 warehouse**, **1 product**, **stable demand**, no delays
- Max steps: 20 | Baseline score: 0.5680

### 🟡 Medium
- **1 warehouse**, **5 products**, **seasonal demand**, 3 suppliers
- Max steps: 40 | Baseline score: 0.4208

### 🔴 Hard
- **3 warehouses**, **10 products**, **noisy seasonal demand + random supplier delays**
- Max steps: 60 | Baseline score: 0.3448

---

## 🔁 Action Space
```python
class SupplyChainAction(Action):
    orders: List[OrderItem]

class OrderItem(State):
    product_id: int     # which product (0 to num_products-1)
    quantity: int       # units to order (0–500)
    supplier_id: int    # 0 = expensive+fast, last = cheap+slow
```

**Example:**
```python
action = SupplyChainAction(orders=[
    OrderItem(product_id=0, quantity=50, supplier_id=1)
])
```

---

## 👁️ Observation Space
```python
class WarehouseObservation(Observation):
    inventory: List[List[float]]          # [num_warehouses x num_products]
    in_transit: List[List[float]]         # arriving next step
    demand_forecast: List[List[float]]    # noisy forecast for next step
    supplier_prices: List[List[float]]    # [num_products x num_suppliers]
    supplier_lead_times: List[List[int]]  # [num_products x num_suppliers]
    stockouts: List[List[float]]          # unmet demand this step
    step_cost: float                      # total cost this step
    reward: float                         # partial progress [-1.0, 1.0]
    done: bool                            # episode ended?
```

---

## 💰 Reward Function
```
reward = clamp(
    service_level × 0.9 − stockout_severity × 0.5 − holding_penalty − 0.05,
    −1.0, +1.0
)
```

| Component | Effect |
|---|---|
| `service_level` | fulfilled / total demand (0–1) |
| `stockout_severity` | unmet demand / total demand (0–1) |
| `holding_penalty` | mild penalty for excess inventory |

Partial progress is built-in — even partially fulfilling demand gives positive reward.

---

## 🚀 Setup
```bash
git clone https://github.com/alwaysgodly/supply-chain-env
cd supply-chain-env
pip install -e .
```

### Run Baseline Agent
```bash
python scripts/baseline_inference.py
python scripts/baseline_inference.py --task easy --display
```

### Run Graders
```bash
python tasks/graders.py --task all --seed 42 --verbose
```

---

## 🐳 Docker
```bash
docker build -t supply-chain-env .
docker run -p 7860:7860 -e TASK_ID=easy supply-chain-env
```

---

## 🤗 Live on HuggingFace Spaces

Try it live: [alwaysgodly/supplychain](https://huggingface.co/spaces/alwaysgodly/supplychain)
```python
from openenv import SyncEnvClient
from supply_chain_env import SupplyChainAction, OrderItem

client = SyncEnvClient(
    url="https://alwaysgodly-supplychain.hf.space",
    action_type=SupplyChainAction,
)
obs = client.reset()
action = SupplyChainAction(orders=[
    OrderItem(product_id=0, quantity=50, supplier_id=1)
])
obs = client.step(action)
print(obs.reward)
```

---

## 📊 Baseline Scores (Greedy Agent, seed=42)
```
easy      [█████████████████░░░░░░░░░░░░░]  0.5680
medium    [████████████░░░░░░░░░░░░░░░░░░]  0.4208
hard      [██████████░░░░░░░░░░░░░░░░░░░░]  0.3448
```

Agent: Greedy Restock (safety_factor=1.5, cheapest supplier). Scores are fully deterministic given the same seed.

---

## 📁 Structure
```
supply-chain-env/
├── src/supply_chain_env/
│   ├── env.py                    # Core environment
│   ├── server.py                 # OpenEnv HTTP server
│   └── __init__.py
├── tasks/
│   └── graders.py                # 3 task graders (easy/medium/hard)
├── scripts/
│   └── baseline_inference.py     # Reproducible baseline scores
├── openenv.yaml                  # OpenEnv spec
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🌐 OpenEnv API Endpoints

Once running (locally or on HuggingFace), the server exposes:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/reset` | POST | Reset environment |
| `/step` | POST | Take an action |
| `/state` | GET | Get internal state |
| `/docs` | GET | Interactive Swagger UI |

---

## 📜 License
MIT
