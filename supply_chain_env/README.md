# 📦 SupplyChainEnv

> A real-world **supply chain restocking** environment for RL agents, built on the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) framework.

The agent manages inventory across warehouses — deciding **what to order**, **how much**, and **from which supplier** — balancing cost efficiency against stockout risk and supplier lead times.

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
- Max steps: 20 | Target score: ≥ 0.70

### 🟡 Medium
- **1 warehouse**, **5 products**, **seasonal demand**, 3 suppliers
- Max steps: 40 | Target score: ≥ 0.50

### 🔴 Hard
- **3 warehouses**, **10 products**, **noisy seasonal demand + random supplier delays**
- Max steps: 60 | Target score: ≥ 0.35

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
    (fulfilled_demand × 2.0 - order_cost - holding_cost - stockout_penalty) / scale,
    -1.0, 1.0
)
```

| Component | Effect |
|---|---|
| `fulfilled_demand × 2.0` | Positive for meeting demand |
| `order_cost` | Price × quantity ordered |
| `holding_cost` | 1% of total inventory per step |
| `stockout_penalty` | 3× unmet demand (strong incentive) |

---

## 🚀 Setup

```bash
git clone https://github.com/your-username/supply-chain-env
cd supply-chain-env
pip install -r requirements.txt
```

### Run Baseline Agent
```bash
python scripts/baseline_inference.py
python scripts/baseline_inference.py --task easy --display
```

### Run Graders
```bash
python tasks/graders.py --task all --verbose
```

---

## 🐳 Docker

```bash
docker build -t supply-chain-env .
docker run -p 8000:8000 -e TASK_ID=easy supply-chain-env
```

---

## 🤗 Hugging Face Spaces

```python
from openenv import SyncEnvClient
from supply_chain_env import SupplyChainAction, OrderItem

client = SyncEnvClient(
    url="https://your-username-supply-chain-env.hf.space",
    action_type=SupplyChainAction,
)
obs = client.reset()
action = SupplyChainAction(orders=[OrderItem(product_id=0, quantity=50, supplier_id=1)])
obs = client.step(action)
```

---

## 📊 Baseline Scores (Greedy Agent, seed=42)

| Task | Score |
|---|---|
| easy | ~0.72 |
| medium | ~0.55 |
| hard | ~0.38 |

---

## 📁 Structure

```
supply-chain-env/
├── src/supply_chain_env/
│   ├── env.py            # Core environment
│   ├── server.py         # OpenEnv HTTP server
│   └── __init__.py
├── tasks/graders.py      # 3 task graders (easy/medium/hard)
├── scripts/baseline_inference.py
├── openenv.yaml          # OpenEnv spec
├── Dockerfile
└── README.md
```

---

## 📜 License
MIT
