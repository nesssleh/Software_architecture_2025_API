"""Policy inference API (FastAPI).

GET  /health
GET  /v1/policies                 — доступные политики и сценарии
POST /v1/act                      — действие по наблюдению
POST /v1/episodes/run             — прогнать эпизод на сервере (опционально PNG последнего кадра)
GET  /v1/stats                    — агрегаты по эпизодам из PostgreSQL
"""
from __future__ import annotations

import base64
import io
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from drone_rl.obs import instrument
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from drone_rl.env.commands import ACTION_NAMES
from drone_rl.env.drone_env import DroneCombatEnv
from drone_rl.env.scenarios import ScenarioFactory
from drone_rl.policies.strategies import PolicyStrategy, make_policy
from drone_rl.storage import EpisodeRepository

MODEL_PATH = os.getenv("MODEL_PATH", "models/ppo_drone_best.zip")
app = FastAPI(title="Drone Combat RL — Policy Inference", version="1.0.0")
instrument(app)


@lru_cache(maxsize=8)
def get_policy(name: str) -> PolicyStrategy:
    if name == "ppo" and not Path(MODEL_PATH).exists():
        raise HTTPException(503, f"model not found: {MODEL_PATH}")
    try:
        return make_policy(name, MODEL_PATH)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@lru_cache(maxsize=1)
def repo() -> EpisodeRepository:
    return EpisodeRepository()


class ActRequest(BaseModel):
    observation: list[float] = Field(..., min_length=17, max_length=17)
    policy: str = "ppo"


class ActResponse(BaseModel):
    action: int
    action_name: str
    policy: str


class EpisodeRequest(BaseModel):
    policy: str = "ppo"
    scenario: str = "mixed"
    seed: int = 0
    render: bool = False


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": Path(MODEL_PATH).exists()}


@app.get("/v1/policies")
def policies():
    return {"policies": ["ppo", "heuristic", "random"], "scenarios": ScenarioFactory.available(), "actions": ACTION_NAMES}


@app.post("/v1/act", response_model=ActResponse)
def act(req: ActRequest):
    p = get_policy(req.policy)
    a = p.act(np.asarray(req.observation, dtype=np.float32))
    return ActResponse(action=a, action_name=ACTION_NAMES[a], policy=p.name)


@app.post("/v1/episodes/run")
def run_episode(req: EpisodeRequest):
    p = get_policy(req.policy)
    try:
        env = DroneCombatEnv(scenario=req.scenario)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    obs, _ = env.reset(seed=req.seed)
    done, actions = False, []
    while not done:
        a = p.act(obs, env)
        actions.append(ACTION_NAMES[a])
        obs, _, term, trunc, info = env.step(a)
        done = term or trunc
    res = {**info["summary"], "policy": p.name, "episode_id": info["episode_id"], "actions": actions}
    repo().upsert(p.name, f"{p.name}-{req.scenario}-{req.seed}", info["summary"])  # идемпотентно по (политика, сценарий, seed)
    if req.render:
        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(env.render()).save(buf, format="PNG")
        res["frame_png_base64"] = base64.b64encode(buf.getvalue()).decode()
    return res


@app.get("/v1/stats")
def stats():
    return {"total_episodes": repo().count(), "by_policy": repo().stats()}
