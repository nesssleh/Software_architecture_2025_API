"""Итоговая оценка PPO против baseline'ов + графики + прогон пайплайна телеметрии.

python scripts/evaluate_and_plot.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from drone_rl.env.drone_env import DroneCombatEnv  # noqa: E402
from drone_rl.evaluate import evaluate  # noqa: E402
from drone_rl.policies.strategies import HeuristicPolicy, PPOPolicy, RandomPolicy  # noqa: E402
from drone_rl.storage import EpisodeRepository  # noqa: E402
from drone_rl.telemetry.kafka_io import InMemoryBroker  # noqa: E402
from services.simulator.run import simulate  # noqa: E402
from services.telemetry_consumer import consumer as cons  # noqa: E402

M, IMG = Path("models"), Path("docs/img")
N = 300


def main() -> None:
    IMG.mkdir(parents=True, exist_ok=True)
    ppo = PPOPolicy(M / "ppo_drone_best.zip")
    res = {}
    for sc in ("open", "forest", "corridor", "mixed"):
        res[sc] = {p.name: evaluate(p, N, scenario=sc) for p in (RandomPolicy(), HeuristicPolicy(), ppo)}
        print(sc, {k: (round(v["success_rate"], 3), round(v["mean_return"], 2)) for k, v in res[sc].items()}, flush=True)

    # training curve
    rows = list(csv.DictReader(open(M / "train_episodes.csv")))
    t = np.array([int(r["timesteps"]) for r in rows])
    ret = np.array([float(r["return"]) for r in rows])
    suc = np.array([int(r["success"]) for r in rows])
    w = 200
    k = np.ones(w) / w
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(t, ret, ",", alpha=0.15, color="#999")
    ax[0].plot(t[w - 1:], np.convolve(ret, k, "valid"), color="#1f5fa8", label=f"скользящее среднее ({w} эп.)")
    h = res["mixed"]["heuristic"]["mean_return"]
    ax[0].axhline(h, color="#d4862a", ls="--", label=f"heuristic baseline ({h:.1f})")
    ax[0].axhline(res["mixed"]["random"]["mean_return"], color="#c33", ls=":", label="random baseline")
    ax[0].set(title="PPO: return за эпизод (обучение)", xlabel="timesteps", ylabel="return")
    ax[0].legend(fontsize=8)
    ax[1].plot(t[w - 1:], np.convolve(suc, k, "valid"), color="#2a8a3e", label="train success rate")
    ev = json.loads((M / "eval_history.json").read_text())
    ax[1].plot([e["timesteps"] for e in ev], [e["success_rate"] for e in ev], "o-", color="#1f5fa8", ms=4, label="eval success (50 эп.)")
    ax[1].plot([e["timesteps"] for e in ev], [e["collision_rate"] for e in ev], "s--", color="#c33", ms=3, label="eval collision")
    ax[1].axhline(res["mixed"]["heuristic"]["success_rate"], color="#d4862a", ls="--", label="heuristic success")
    ax[1].set(title="Доля успешных эпизодов", xlabel="timesteps", ylim=(0, 1.02))
    ax[1].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(IMG / "training_curve.png", dpi=130)
    plt.close(fig)

    # сравнение по сценариям
    fig, ax = plt.subplots(figsize=(9, 3.8))
    scs, pols = list(res), ["random", "heuristic", "ppo"]
    x = np.arange(len(scs))
    for i, p in enumerate(pols):
        ax.bar(x + (i - 1) * 0.27, [res[s][p]["success_rate"] for s in scs], 0.27, label=p)
    ax.set_xticks(x, scs)
    ax.set(title=f"Success rate по сценариям ({N} эпизодов, одинаковые seed)", ylim=(0, 1.05))
    ax.legend()
    ax.grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(IMG / "success_by_scenario.png", dpi=130)
    plt.close(fig)

    # кадры эпизодов PPO
    frames = []
    for sc, seed in (("forest", 3), ("corridor", 5), ("mixed", 11)):
        env = DroneCombatEnv(scenario=sc)
        obs, _ = env.reset(seed=seed)
        done = False
        while not done:
            obs, _, te, tr, info = env.step(ppo.act(obs))
            done = te or tr
        frames.append((sc, info["summary"], env.render()))
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.6))
    for a, (sc, s, fr) in zip(ax, frames):
        a.imshow(fr)
        a.set_title(f"{sc}: {s['outcome']}, {s['steps']} шагов, попаданий {s['hits']}/{s['shots']}", fontsize=9)
        a.axis("off")
    fig.tight_layout()
    fig.savefig(IMG / "episodes_ppo.png", dpi=110)
    plt.close(fig)

    # пайплайн телеметрии: simulator -> Kafka -> consumer -> PostgreSQL/SQLite
    broker = InMemoryBroker(partitions=6)
    repo = EpisodeRepository("sqlite:///models/demo_episodes.db")
    for pol in ("random", "heuristic", "ppo"):
        simulate(pol, 40, "mixed", str(M / "ppo_drone_best.zip"), producer=broker.producer(), seed0=777)
    n = cons.run(repo, broker.consumer("drone.episodes", group_id="episode-writer"), broker.producer(), once=True)
    pipe = {"telemetry_msgs": len(broker.messages("drone.telemetry")), "episode_msgs": len(broker.messages("drone.episodes")),
            "consumed": n, "lag": broker.lag("episode-writer", "drone.episodes"), "db_stats": repo.stats()}
    info = json.loads((M / "train_info.json").read_text())
    (M / "final_eval.json").write_text(json.dumps({"episodes_per_cell": N, "results": res, "pipeline": pipe, "train": info},
                                                 indent=1, ensure_ascii=False))
    print(json.dumps(pipe, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
