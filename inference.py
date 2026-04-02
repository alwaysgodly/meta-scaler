"""
Inference Script — SupplyChainEnv
===================================
Uses an LLM agent (via OpenAI client) to interact with SupplyChainEnv.
Emits structured [START] / [STEP] / [END] logs to stdout.

Required environment variables:
    API_BASE_URL     The API endpoint for the LLM.
    MODEL_NAME       The model identifier to use for inference.
    HF_TOKEN         Your Hugging Face / API key.
    TASK_NAME        Task to run: easy | medium | hard (default: easy)
"""

import os
import json
import sys
from typing import List, Optional

from openai import OpenAI

# --- env vars ---
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME = os.getenv("TASK_NAME", "easy")
MAX_STEPS = 60
TEMPERATURE = 0.2
MAX_TOKENS = 512

# add src and tasks to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tasks"))

from supply_chain_env.env import SupplyChainEnv, SupplyChainAction, OrderItem, TASK_CONFIGS


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)


# ---------------------------------------------------------------------------
# LLM agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a supply chain inventory management agent.
You will receive the current warehouse state and must decide what to order.

You must respond with a JSON object in this exact format:
{
  "orders": [
    {"product_id": 0, "quantity": 50, "supplier_id": 1},
    {"product_id": 1, "quantity": 30, "supplier_id": 2}
  ]
}

Rules:
- Only include products that need restocking (quantity > 0)
- quantity must be between 0 and 500
- supplier_id: 0 = expensive+fast, last index = cheapest+slowest
- Order enough to cover forecast demand with a safety buffer
- Respond with valid JSON only, no explanation
"""


def build_prompt(obs, cfg: dict) -> str:
    NW = len(obs.inventory)
    NP = len(obs.inventory[0]) if NW > 0 else 0
    NS = len(obs.supplier_prices[0]) if NP > 0 else 0

    return f"""Current warehouse state:
- Inventory: {obs.inventory}
- In-transit (arriving next step): {obs.in_transit}
- Demand forecast (noisy): {obs.demand_forecast}
- Supplier prices [product x supplier]: {obs.supplier_prices}
- Supplier lead times [product x supplier]: {obs.supplier_lead_times}
- Last stockouts: {obs.stockouts}
- Last step cost: {obs.step_cost}
- Last reward: {obs.reward}

Task: {TASK_NAME} | Warehouses: {NW} | Products: {NP} | Suppliers: {NS}

Decide which products to restock, how much, and from which supplier.
Respond with JSON only."""


def get_llm_action(client: OpenAI, obs, cfg: dict) -> SupplyChainAction:
    """Ask the LLM what orders to place."""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(obs, cfg)},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        response_text = completion.choices[0].message.content or "{}"

        # strip markdown code fences if present
        response_text = response_text.strip()
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
        response_text = response_text.strip()

        data = json.loads(response_text)
        orders = []
        for o in data.get("orders", []):
            orders.append(OrderItem(
                product_id=int(o["product_id"]),
                quantity=int(o["quantity"]),
                supplier_id=int(o["supplier_id"]),
            ))
        return SupplyChainAction(orders=orders)

    except Exception as e:
        # fallback: greedy restock
        return fallback_action(obs)


def fallback_action(obs) -> SupplyChainAction:
    """Simple greedy fallback if LLM fails."""
    orders = []
    NP = len(obs.inventory[0]) if obs.inventory else 0
    NS = len(obs.supplier_prices[0]) if obs.supplier_prices else 1

    for p in range(NP):
        total_inv = sum(obs.inventory[w][p] for w in range(len(obs.inventory)))
        total_transit = sum(obs.in_transit[w][p] for w in range(len(obs.in_transit)))
        total_forecast = sum(obs.demand_forecast[w][p] for w in range(len(obs.demand_forecast)))

        gap = max(0.0, total_forecast * 1.5 - total_inv - total_transit)
        if gap > 1.0:
            orders.append(OrderItem(
                product_id=p,
                quantity=int(round(gap)),
                supplier_id=NS - 1,
            ))
    return SupplyChainAction(orders=orders)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    cfg = TASK_CONFIGS[TASK_NAME]

    env = SupplyChainEnv(task_id=TASK_NAME, seed=42)

    rewards: List[float] = []
    steps_taken = 0
    success = False

    log_start(task=TASK_NAME, env="supply-chain-env", model=MODEL_NAME)

    try:
        obs = env.reset(seed=42)

        for step in range(1, MAX_STEPS + 1):
            if obs.done:
                break

            # get action from LLM
            action = get_llm_action(client, obs, cfg)
            action_str = json.dumps([
                {"p": o.product_id, "q": o.quantity, "s": o.supplier_id}
                for o in action.orders
            ])

            # step the environment
            try:
                obs = env.step(action)
                reward = obs.reward if obs.reward is not None else 0.0
                done = obs.done
                error = None
            except Exception as e:
                reward = 0.0
                done = True
                error = str(e)

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                success = reward > 0.0
                break

        else:
            success = False

    except Exception as e:
        print(f"[DEBUG] Fatal error: {e}", flush=True)
        success = False

    finally:
        log_end(success=success, steps=steps_taken, rewards=rewards)


if __name__ == "__main__":
    main()
