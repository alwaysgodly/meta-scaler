import os
from supply_chain_env.env import SupplyChainEnv, SupplyChainAction, WarehouseObservation
from openenv.core import create_fastapi_app
from fastapi.responses import RedirectResponse

TASK_ID = os.environ.get("TASK_ID", "easy")

def make_env() -> SupplyChainEnv:
    return SupplyChainEnv(task_id=TASK_ID)

app = create_fastapi_app(
    env=make_env,
    action_cls=SupplyChainAction,
    observation_cls=WarehouseObservation,
)

@app.get("/")
def root():
    return RedirectResponse(url="/docs")
