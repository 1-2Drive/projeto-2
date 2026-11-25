"""
Avaliação do planeador A* em todas as cenas, com validação cruzada por cenário.

O `run_predict_to_plan.py` testa o planeador nas primeiras 30 situações do
conjunto de teste, que vêm de meia dúzia de cenas. Este script usa todas as
situações de todas as cenas: em cada parte da validação cruzada, a floresta
aleatória é treinada nas outras partes, por isso nenhuma cena é planeada por
um modelo que a viu no treino.

Compara quatro fontes de previsão para os obstáculos:
  nenhuma  - os outros veículos ficam onde estão;
  cv       - velocidade constante;
  rf       - floresta aleatória (referencial do veículo + correção);
  oraculo  - o futuro real (limite superior).

Os intervalos de confiança são calculados por reamostragem das cenas
(bootstrap por cenário), porque as situações da mesma cena não são
independentes.

Uso:
    python scripts/evaluate_planner_cv.py
    python scripts/evaluate_planner_cv.py --csv data/processed/predict_to_plan_samples.csv

Requer o CSV de amostras gerado por `preprocess_at_scale.py`
(com as colunas `lead_*` e `future_agent_*`).
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.trajectory_predictor import MLTrajectoryPredictor
from src.planning.astar_planner import AStarTrajectoryPlanner

MODES = ["nenhuma", "cv", "rf", "oraculo"]
HORIZON_STEPS = 30  # 3,0 s a 10 Hz


def is_interactive_scene(group: pd.DataFrame) -> bool:
    """Mesmo critério do run_predict_to_plan.py: há um veículo à frente do ego, a menos de 25 m."""
    return bool(((group.distance_to_ego < 25) & (group.rel_y.abs() < 4) & (group.rel_x > 1)).any())


def obstacle_trajectories(group: pd.DataFrame, pred_3s: np.ndarray) -> dict:
    """Posições dos obstáculos, passo a passo, para cada fonte de previsão."""
    steps = range(HORIZON_STEPS + 1)
    out = {mode: {} for mode in MODES}
    for k, (_, o) in enumerate(group.iterrows()):
        tid = int(o.track_id)
        out["nenhuma"][tid] = [(o.agent_x, o.agent_y)] * (HORIZON_STEPS + 1)
        out["cv"][tid] = [(o.agent_x + o.agent_vx * j * 0.1, o.agent_y + o.agent_vy * j * 0.1) for j in steps]
        out["rf"][tid] = [
            (o.agent_x + pred_3s[k, 0] * j / HORIZON_STEPS, o.agent_y + pred_3s[k, 1] * j / HORIZON_STEPS)
            for j in steps
        ]
        out["oraculo"][tid] = [
            (o.agent_x + o.future_dx_3s * j / HORIZON_STEPS, o.agent_y + o.future_dy_3s * j / HORIZON_STEPS)
            for j in steps
        ]
    return out


def run(df: pd.DataFrame, n_splits: int) -> pd.DataFrame:
    planner = AStarTrajectoryPlanner(horizon_steps=HORIZON_STEPS, dt=0.1, max_speed=18.0, safety_radius=3.5)
    rows = []
    t0 = time.time()
    folds = GroupKFold(n_splits=n_splits).split(df, groups=df.scenario_id)
    for fold, (idx_train, idx_test) in enumerate(folds):
        train, test = df.iloc[idx_train], df.iloc[idx_test]
        rf = MLTrajectoryPredictor(model_type="random_forest", n_estimators=60, max_depth=8).fit(train)
        for (scenario_id, step), group in test.groupby(["scenario_id", "current_time_step"]):
            if not is_interactive_scene(group):
                continue
            ego = group.iloc[0]
            s_pts = np.linspace(0, 60, 60)
            path = np.column_stack(
                [ego.ego_x + s_pts * np.cos(ego.ego_heading), ego.ego_y + s_pts * np.sin(ego.ego_heading)]
            )
            obstacles = obstacle_trajectories(group, rf.predict_displacements(group)["3s"])
            start_gap = float(np.hypot(group.agent_x - ego.ego_x, group.agent_y - ego.ego_y).min())
            headings = {int(o.track_id): float(o.agent_heading) for _, o in group.iterrows()}
            row = {"scenario_id": scenario_id, "step": int(step), "fold": fold, "start_gap": start_gap}
            for mode in MODES:
                s_tr, v_tr, a_tr = planner.plan_trajectory(float(ego.ego_speed), path, obstacles[mode])
                res = planner.evaluate_plan_against_ground_truth(
                    mode, path, s_tr, v_tr, a_tr, obstacles["oraculo"],
                    ego_heading=float(ego.ego_heading), gt_headings=headings,
                )
                row[f"{mode}_conflict"] = bool(res.collision_occurred)  # a menos de 3,5 m, centro a centro
                row[f"{mode}_collision"] = bool(res.obb_collision_occurred)  # sobreposição das caixas dos veículos
                row["start_overlap"] = bool(res.initial_overlap_detected)
                row[f"{mode}_hard_brakes"] = int(res.emergency_braking_events)
                row[f"{mode}_speed"] = float(res.average_speed)
            rows.append(row)
        print(f"  parte {fold + 1}/{n_splits}: {len(rows)} situações ({time.time() - t0:.0f} s)", flush=True)
    return pd.DataFrame(rows)


def bootstrap_by_scenario(results: pd.DataFrame, n_boot: int, seed: int = 0, metric: str = "conflict"):
    """Percentis 2,5 e 97,5 da taxa (conflito ou colisão) de cada modo e das diferenças entre modos."""
    rng = np.random.default_rng(seed)
    by_scenario = {s: g.index.to_numpy() for s, g in results.groupby("scenario_id")}
    keys = list(by_scenario)
    pairs = [(a, b) for i, a in enumerate(MODES) for b in MODES[i + 1 :]]
    rates = {m: [] for m in MODES}
    diffs = {p: [] for p in pairs}
    for _ in range(n_boot):
        sample = results.loc[np.concatenate([by_scenario[k] for k in rng.choice(keys, len(keys))])]
        for m in MODES:
            rates[m].append(sample[f"{m}_{metric}"].mean() * 100)
        for a, b in pairs:
            diffs[(a, b)].append((sample[f"{a}_{metric}"].mean() - sample[f"{b}_{metric}"].mean()) * 100)

    def ci(values):
        return tuple(np.percentile(values, [2.5, 97.5]))

    return {m: ci(v) for m, v in rates.items()}, {p: ci(v) for p, v in diffs.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default="data/processed/predict_to_plan_samples.csv")
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--out", default="data/processed/planner_cv_results.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    missing = [c for c in ("lead_gap", "future_agent_dx_3s") if c not in df.columns]
    if missing:
        sys.exit(f"O CSV não tem as colunas {missing}: volte a gerar as amostras com preprocess_at_scale.py.")

    print(f"Amostras: {len(df)} | cenas: {df.scenario_id.nunique()}")
    results = run(df, args.splits)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out, index=False)

    print(f"\nSituações: {len(results)} | cenas: {results.scenario_id.nunique()}")
    print(f"Situações que já começam com um veículo a menos de 3,5 m: {(results.start_gap < 3.5).mean() * 100:.1f}%")
    print(f"Situações que já começam com as caixas sobrepostas: {results.start_overlap.mean() * 100:.1f}%")

    labels = {"collision": "colisão (caixas dos veículos sobrepostas)", "conflict": "conflito (a menos de 3,5 m, centro a centro)"}
    for metric, label in labels.items():
        rate_ci, diff_ci = bootstrap_by_scenario(results, args.bootstrap, metric=metric)
        print(f"\n== {label}")
        print(f"{'previsão':<10}{'%':>8}{'IC 95%':>16}{'travagens máx.':>16}{'vel. média':>12}")
        for m in MODES:
            lo, hi = rate_ci[m]
            print(
                f"{m:<10}{results[f'{m}_{metric}'].mean() * 100:>8.1f}{f'[{lo:.1f}; {hi:.1f}]':>16}"
                f"{results[f'{m}_hard_brakes'].mean():>16.2f}{results[f'{m}_speed'].mean():>12.2f}"
            )
        print("Diferenças (pontos percentuais; IC 95% por cenário):")
        for (a, b), (lo, hi) in diff_ci.items():
            d = (results[f"{a}_{metric}"].mean() - results[f"{b}_{metric}"].mean()) * 100
            flag = "" if lo <= 0 <= hi else "  <- o intervalo exclui o zero"
            print(f"  {a} - {b}: {d:+.1f} [{lo:+.1f}; {hi:+.1f}]{flag}")
    print(f"\nResultados por situação em {args.out}")


if __name__ == "__main__":
    main()
