"""
Inference Script — SupplyChainEnv (Fixed for Meta PyTorch Hackathon)
===================================================================
Added: FastAPI wrapper to prevent container exit and handle health checks.
"""

import os
import json
import sys
import threading
from typing import List, Optional

# Web Server Imports
from fastapi import FastAPI
import uvicorn

# --- env vars ---
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "dummy-key"
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME = os.getenv("TASK_NAME", "easy")
MAX_STEPS = 60
TEMPERATURE = 0.2
MAX_TOKENS = 512

# Fix paths based on your supply_chain_env folder structure
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "supply_chain_env"))

from supply_chain_env.env import SupplyChainEnv, SupplyChainAction, OrderItem, TASK_CONFIGS

# ---------------------------------------------------------------------------
# FastAPI Setup
# ---------------------------------------------------------------------------
app = FastAPI()

@app.get("/")
@app.get("/health")
def health_check():
    """Satisfies the validator's health check."""
    return {"status": "healthy", "task": TASK_NAME}

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)

# ---------------------------------------------------------------------------
# Fallback & LLM Logic (Original)
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
            orders.append(OrderItem(product_id=p, quantity=int(round(gap)), supplier_id=NS - 1))
    return SupplyChainAction(orders=orders)

SYSTEM_PROMPT = """You are a supply chain inventory management agent. Respond with JSON object only."""

def get_llm_action(client, obs, cfg: dict) -> SupplyChainAction:
    try:
        NW = len(obs.inventory)
        NP = len(obs.inventory[0]) if NW > 0 else 0
        NS = len(obs.supplier_prices[0]) if NP > 0 else 1
        prompt = f"Inventory: {obs.inventory}\nTask: {TASK_NAME}"
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            temperature=TEMPERATURE, max_tokens=MAX_TOKENS, timeout=10,
        )
        response_text = completion.choices[0].message.content or "{}"
        data = json.loads(response_text.strip("`").replace("json", ""))
        orders = [OrderItem(product_id=int(o["product_id"]), quantity=min(500, max(0, int(o["quantity"]))), supplier_id=int(o["supplier_id"])) for o in data.get("orders", [])]
        return SupplyChainAction(orders=orders)
    except Exception:
        return fallback_action(obs)

# ---------------------------------------------------------------------------
# Main Simulation Loop
# ---------------------------------------------------------------------------

def run_simulation() -> None:
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
            client, use_llm = None, False

        cfg = TASK_CONFIGS[TASK_NAME]
        env = SupplyChainEnv(task_id=TASK_NAME, seed=42)
        obs = env.reset(seed=42)

        for step in range(1, cfg["max_steps"] + 1):
            if obs.done: break
            action = get_llm_action(client, obs, cfg) if use_llm else fallback_action(obs)
            action_str = json.dumps([{"p": o.product_id, "q": o.quantity, "s": o.supplier_id} for o in action.orders])
            try:
                obs = env.step(action)
                reward, done, error = float(obs.reward or 0.0), bool(obs.done), None
            except Exception as e:
                reward, done, error = 0.0, True, str(e)[:100]
            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)
            if done:
                success = reward > 0.0
                break
    except Exception as e:
        print(f"[DEBUG] Fatal: {e}", flush=True)
    finally:
        log_end(success=success, steps=steps_taken, rewards=rewards)

# ---------------------------------------------------------------------------
# Execution Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. Run simulation in background
    sim_thread = threading.Thread(target=run_simulation, daemon=True)
    sim_thread.start()
    
    # 2. Run server in foreground to stay "Healthy"
    # Port 7860 is mandatory based on your logs
    uvicorn.run(app, host="0.0.0.0", port=7860)
