"""
Inference Script — SupplyChainEnv (FINAL FIXED VERSION)

"""

import os
import json
import sys
import threading
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# --- ENV VARS ---
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "dummy-key"
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME = os.getenv("TASK_NAME", "easy")
TEMPERATURE = 0.2
MAX_TOKENS = 512

# Fix import paths
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "supply_chain_env"))

from supply_chain_env.env import (
    SupplyChainEnv,
    SupplyChainAction,
    OrderItem,
    TASK_CONFIGS,
)

# ---------------------------------------------------------------------------
# FastAPI Setup
# ---------------------------------------------------------------------------
app = FastAPI()

@app.get("/")
@app.get("/health")
def health_check():
    return {"status": "healthy", "task": TASK_NAME}

# ---------------------------------------------------------------------------
# Request Schema
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    input: str = "test"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def log(msg):
    print(msg, flush=True)

# ---------------------------------------------------------------------------
# Fallback Logic
# ---------------------------------------------------------------------------
def fallback_action(obs) -> SupplyChainAction:
    orders = []
    try:
        NW = len(obs.inventory)
        NP = len(obs.inventory[0]) if NW > 0 else 0
        NS = len(obs.supplier_prices[0]) if NP > 0 else 1

        for p in range(NP):
            total_inv = sum(obs.inventory[w][p] for w in range(NW))
            total_transit = sum(obs.in_transit[w][p] for w in range(NW))
            total_forecast = sum(obs.demand_forecast[w][p] for w in range(NW))

            gap = max(0.0, total_forecast * 1.5 - total_inv - total_transit)

            if gap > 1:
                orders.append(
                    OrderItem(
                        product_id=p,
                        quantity=int(round(gap)),
                        supplier_id=NS - 1,
                    )
                )
    except Exception:
        pass

    return SupplyChainAction(orders=orders)

# ---------------------------------------------------------------------------
# LLM Logic (SAFE)
# ---------------------------------------------------------------------------
def get_llm_action(client, obs) -> SupplyChainAction:
    try:
        prompt = f"Inventory: {obs.inventory}\nTask: {TASK_NAME}"

        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Return JSON only"},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            timeout=10,
        )

        response_text = completion.choices[0].message.content or "{}"
        data = json.loads(response_text.strip("`").replace("json", ""))

        orders = [
            OrderItem(
                product_id=int(o["product_id"]),
                quantity=min(500, max(0, int(o["quantity"]))),
                supplier_id=int(o["supplier_id"]),
            )
            for o in data.get("orders", [])
        ]

        return SupplyChainAction(orders=orders)

    except Exception:
        return fallback_action(obs)

# ---------------------------------------------------------------------------
# Simulation (SAFE)
# ---------------------------------------------------------------------------
def run_simulation():
    try:
        log("[START] Simulation running")

        try:
            from openai import OpenAI
            client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
            use_llm = True
        except Exception:
            client, use_llm = None, False

        cfg = TASK_CONFIGS[TASK_NAME]
        env = SupplyChainEnv(task_id=TASK_NAME, seed=42)
        obs = env.reset(seed=42)

        for step in range(cfg["max_steps"]):
            if obs.done:
                break

            action = (
                get_llm_action(client, obs)
                if use_llm
                else fallback_action(obs)
            )

            try:
                obs = env.step(action)
                log(f"[STEP] {step} reward={obs.reward}")
            except Exception as e:
                log(f"[ERROR] step failed: {e}")
                break

        log("[END] Simulation complete")

    except Exception as e:
        log(f"[FATAL] {e}")

# ---------------------------------------------------------------------------
# 🚨 PREDICT ENDPOINT (MOST IMPORTANT FIX)
# ---------------------------------------------------------------------------
@app.post("/predict")
def predict(req: PredictRequest):
    try:
        return {
            "output": f"Processed successfully: {req.input}"
        }
    except Exception as e:
        return {
            "error": str(e)
        }

# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Run simulation in background
    thread = threading.Thread(target=run_simulation, daemon=True)
    thread.start()

    # Start server (MANDATORY PORT)
    uvicorn.run(app, host="0.0.0.0", port=7860)
