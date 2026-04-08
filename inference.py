"""
Inference Script — SupplyChainEnv
Required env vars:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    TASK_NAME      Task to run: easy | medium | hard (default: easy)
"""

import os
import json
import sys
from typing import List, Optional

# Fix import paths — works both locally and in Docker (/app)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
sys.path.insert(0, os.path.join(BASE_DIR, "tasks"))
sys.path.insert(0, "/app/src")
sys.path.insert(0, "/app/tasks")

# --- env vars ---
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "dummy-key"
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME = os.getenv("TASK_NAME", "easy")
MAX_STEPS = 60
TEMPERATURE = 0.2
MAX_TOKENS = 512

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
# Fallback greedy policy
# ---------------------------------------------------------------------------

def fallback_action(obs) -> SupplyChainAction:
    orders = []
    NW = len(obs.inventory) if obs.inventory else 0
    NP = len(obs.inventory[0]) if NW > 0 else 0
    NS = len(obs.supplier_prices[0]) if NP > 0 and obs.supplier_prices else 1

    for p in range(NP):
        total_inv = sum(obs.inventory[w][p] for w in range(NW))
        total_transit = sum(obs.in_transit[w][p] for w in range(NW))
        total_forecast = sum(obs.demand_forecast[w][p] for w in range(NW))
        gap = max(0.0, total_forecast * 1.5 - total_inv - total_transit)
        if gap > 1.0:
            orders.append(OrderItem(
                product_id=p,
                quantity=min(500, max(0, int(round(gap)))),
                supplier_id=NS - 1,
            ))
    return SupplyChainAction(orders=orders)


# ---------------------------------------------------------------------------
# LLM agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a supply chain inventory management agent.
Respond with JSON only:
{"orders": [{"product_id": 0, "quantity": 50, "supplier_id": 1}]}
Rules: quantity 0-500, supplier 0=fast+expensive, last=slow+cheap."""


def get_llm_action(client, obs, cfg: dict) -> SupplyChainAction:
    try:
        NW = len(obs.inventory)
        NP = len(obs.inventory[0]) if NW > 0 else 0
        NS = len(obs.supplier_prices[0]) if NP > 0 else 1

        prompt = f"""Inventory: {obs.inventory}
In-transit: {obs.in_transit}
Demand forecast: {obs.demand_forecast}
Supplier prices: {obs.supplier_prices}
Lead times: {obs.supplier_lead_times}
Stockouts: {obs.stockouts}
Task: {TASK_NAME} | Warehouses: {NW} | Products: {NP} | Suppliers: {NS}
JSON only."""

        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            timeout=10,
        )
        response_text = completion.choices[0].message.content or "{}"
        response_text = response_text.strip()
        if "```" in response_text:
            parts = response_text.split("```")
            response_text = parts[1] if len(parts) > 1 else parts[0]
            if response_text.startswith("json"):
                response_text = response_text[4:]
        response_text = response_text.strip()

        data = json.loads(response_text)
        orders = []
        for o in data.get("orders", []):
            orders.append(OrderItem(
                product_id=int(o["product_id"]),
                quantity=min(500, max(0, int(o["quantity"]))),
                supplier_id=int(o["supplier_id"]),
            ))
        return SupplyChainAction(orders=orders)
    except Exception:
        return fallback_action(obs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    rewards: List[float] = []
    steps_taken = 0
    success = False

    log_start(task=TASK_NAME, env="supply-chain-env", model=MODEL_NAME)

    try:
        try:
            from openai import OpenAI
            client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
            use_llm = True
        except Exception:
            client = None
            use_llm = False

        cfg = TASK_CONFIGS[TASK_NAME]
        env = SupplyChainEnv(task_id=TASK_NAME, seed=42)
        obs = env.reset(seed=42)

        for step in range(1, cfg["max_steps"] + 1):
            if obs.done:
                break

            try:
                if use_llm and client:
                    action = get_llm_action(client, obs, cfg)
                else:
                    action = fallback_action(obs)
            except Exception:
                action = fallback_action(obs)

            action_str = json.dumps([
                {"p": o.product_id, "q": o.quantity, "s": o.supplier_id}
                for o in action.orders
            ])

            try:
                obs = env.step(action)
                reward = float(obs.reward) if obs.reward is not None else 0.0
                done = bool(obs.done)
                error = None
            except Exception as e:
                reward = 0.0
                done = True
                error = str(e)[:100]

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                success = reward > 0.0
                break

        else:
            success = False

    except Exception as e:
        print(f"[DEBUG] Fatal: {e}", flush=True)
        success = False

    finally:
        log_end(success=success, steps=steps_taken, rewards=rewards)


if __name__ == "__main__":
    main()
