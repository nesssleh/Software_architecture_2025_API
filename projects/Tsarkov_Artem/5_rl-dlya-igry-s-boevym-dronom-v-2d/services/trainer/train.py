"""RL Trainer: обучение PPO (Stable-Baselines3) на DroneCombatEnv.

Логирует метрики в MLflow (если доступен) и в CSV, сохраняет лучшую модель
по success_rate на валидационных эпизодах. После обучения публикует событие
model.registered в Kafka-топик drone.models (если задан KAFKA_BOOTSTRAP).

Запуск:  python -m services.trainer.train --timesteps 600000
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from drone_rl.env.drone_env import DroneCombatEnv
from drone_rl.evaluate import evaluate
from drone_rl.policies.strategies import PolicyStrategy

try:
    import mlflow  # type: ignore
except Exception:  # pragma: no cover
    mlflow = None


class _ModelPolicy(PolicyStrategy):
    name = "ppo"

    def __init__(self, model):
        self.model = model

    def act(self, obs, env=None):
        return int(self.model.predict(obs, deterministic=True)[0])


class TrainLogger(BaseCallback):
    def __init__(self, out_dir: Path, eval_every: int, eval_episodes: int = 50):
        super().__init__()
        self.out_dir = out_dir
        self.eval_every = eval_every
        self.eval_episodes = eval_episodes
        self.ep_rows: list[dict] = []
        self.eval_rows: list[dict] = []
        self.best = -1.0
        self._next_eval = eval_every

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "summary" in info:
                s = info["summary"]
                self.ep_rows.append({"timesteps": self.num_timesteps, "return": s["return"],
                                     "success": int(s["success"]), "outcome": s["outcome"], "steps": s["steps"]})
        if self.num_timesteps >= self._next_eval:
            self._next_eval += self.eval_every
            m = evaluate(_ModelPolicy(self.model), self.eval_episodes, seed0=50_000)
            m["timesteps"] = self.num_timesteps
            self.eval_rows.append(m)
            print(f"[eval] t={self.num_timesteps} success={m['success_rate']:.2f} "
                  f"collision={m['collision_rate']:.2f} return={m['mean_return']:.2f}", flush=True)
            if mlflow:
                mlflow.log_metrics({"eval_success_rate": m["success_rate"], "eval_mean_return": m["mean_return"],
                                    "eval_collision_rate": m["collision_rate"]}, step=self.num_timesteps)
            if m["success_rate"] >= self.best:
                self.best = m["success_rate"]
                self.model.save(self.out_dir / "ppo_drone_best")
        return True

    def dump(self) -> None:
        with open(self.out_dir / "train_episodes.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["timesteps", "return", "success", "outcome", "steps"])
            w.writeheader()
            w.writerows(self.ep_rows)
        with open(self.out_dir / "eval_history.json", "w") as f:
            json.dump(self.eval_rows, f, indent=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=600_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--scenario", default="mixed")
    ap.add_argument("--out", default="models")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    env = make_vec_env(lambda: Monitor(DroneCombatEnv(scenario=args.scenario)), n_envs=args.n_envs, seed=args.seed)
    hparams = dict(learning_rate=3e-4, n_steps=512, batch_size=256, n_epochs=10, gamma=0.99,
                   gae_lambda=0.95, clip_range=0.2, ent_coef=0.01, policy_kwargs=dict(net_arch=[128, 128]))
    model = PPO("MlpPolicy", env, seed=args.seed, verbose=0, device="cpu", **hparams)

    if mlflow:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{out.resolve()}/mlflow.db"))
        mlflow.set_experiment("drone-combat-ppo")
        mlflow.start_run()
        mlflow.log_params({**{k: v for k, v in hparams.items() if k != "policy_kwargs"},
                           "net_arch": "128x128", "timesteps": args.timesteps, "n_envs": args.n_envs,
                           "scenario": args.scenario})
    cb = TrainLogger(out, eval_every=max(args.timesteps // 20, 1000))
    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, callback=cb)
    model.save(out / "ppo_drone_last")
    cb.dump()
    info = {"timesteps": args.timesteps, "train_seconds": round(time.time() - t0, 1), "best_eval_success": cb.best,
            "hparams": {k: str(v) for k, v in hparams.items()}}
    (out / "train_info.json").write_text(json.dumps(info, indent=1))
    if mlflow:
        mlflow.log_artifact(str(out / "ppo_drone_best.zip"))
        mlflow.end_run()
    if os.getenv("KAFKA_BOOTSTRAP"):
        from drone_rl.telemetry.kafka_io import make_producer
        make_producer().send("drone.models", key="ppo", value=json.dumps({"event": "model.registered", **info}))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
