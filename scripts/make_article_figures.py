"""
Gera as figuras e os números de previsão do artigo a partir da tabela de amostras.

Tudo é calculado com validação cruzada agrupada por cenário: cada amostra é
prevista por um modelo que nunca viu a sua cena.

Figuras (PDF) escritas em --out:
  erro_horizonte.pdf   erro final (FDE) por horizonte e por modelo
  importancias.pdf     seis variáveis com mais peso na floresta aleatória
  confusao.pdf         matriz de confusão normalizada das manobras
  exemplo.pdf          um caso real: trajetória verdadeira, velocidade constante e floresta aleatória

Uso:
    python scripts/make_article_figures.py
    python scripts/make_article_figures.py --csv data/processed/predict_to_plan_samples.csv --out artigo_final/figuras
"""

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.trajectory_predictor import (
    ConstantAccelerationPredictor,
    ConstantVelocityPredictor,
    ManeuverClassifier,
    MLTrajectoryPredictor,
)

HORIZONS = ["1s", "3s", "5s"]
INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#dddcd5"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

# nome, fábrica, cor, traço, marcador
MODELS = [
    ("Velocidade constante", ConstantVelocityPredictor, MUTED, "--", "o"),
    ("Aceleração constante", ConstantAccelerationPredictor, MUTED, ":", "s"),
    (
        "Floresta aleatória (mapa)",
        lambda: MLTrajectoryPredictor(
            model_type="random_forest", frame="global", residual=False, n_estimators=60, max_depth=8
        ),
        ORANGE,
        "-",
        "^",
    ),
    ("Árvore de decisão", lambda: MLTrajectoryPredictor(model_type="decision_tree", max_depth=6), AQUA, "-", "v"),
    (
        "Floresta aleatória",
        lambda: MLTrajectoryPredictor(model_type="random_forest", n_estimators=60, max_depth=8),
        BLUE,
        "-",
        "D",
    ),
    (
        "Gradient boosting",
        lambda: MLTrajectoryPredictor(model_type="gradient_boosting", max_iter=80, max_depth=5),
        INK,
        "-",
        "x",
    ),
]

FEATURE_NAMES = {
    "agent_accel_est": "aceleração estimada",
    "yaw_rate_1s": "taxa de viragem",
    "lead_closing_speed": "aproximação ao da frente",
    "past_lat_1s": "desvio lateral (1 s)",
    "lead_gap": "distância ao da frente",
    "v_lon": "velocidade longitudinal",
    "v_lat": "velocidade lateral",
    "agent_speed": "velocidade",
    "past_lon_1s": "avanço (1 s)",
    "past_speed_mean": "velocidade média (1 s)",
    "lead_speed": "velocidade do da frente",
    "rel_x": "posição relativa ao ego (frente)",
    "rel_y": "posição relativa ao ego (lado)",
    "rel_vx": "velocidade relativa ao ego (frente)",
    "rel_vy": "velocidade relativa ao ego (lado)",
    "rel_heading": "orientação relativa ao ego",
    "distance_to_ego": "distância ao ego",
    "closing_speed": "aproximação ao ego",
    "ttc": "tempo até à colisão com o ego",
    "azimuth_angle": "ângulo em relação ao ego",
    "ego_speed": "velocidade do ego",
}
MANEUVERS = ["STRAIGHT", "ACCELERATING", "DECELERATING", "TURN_LEFT", "TURN_RIGHT"]
MANEUVER_NAMES = ["em frente", "acelerar", "travar", "virar à esq.", "virar à dir."]


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def out_of_fold(df, n_splits):
    """Erro final por parte e por modelo, importâncias médias e previsões fora da amostra."""
    folds = list(GroupKFold(n_splits=n_splits).split(df, groups=df.scenario_id))
    fde = {name: {h: [] for h in HORIZONS} for name, *_ in MODELS}
    importances = []
    oof = {name: {h: np.full((len(df), 2), np.nan) for h in HORIZONS} for name in ("Velocidade constante", "Floresta aleatória")}
    for idx_train, idx_test in folds:
        train, test = df.iloc[idx_train], df.iloc[idx_test]
        for name, factory, *_ in MODELS:
            model = factory()
            if hasattr(model, "fit"):
                model.fit(train)
            pred = model.predict_displacements(test)
            for h in HORIZONS:
                true = test[[f"future_dx_{h}", f"future_dy_{h}"]].to_numpy()
                err = np.hypot(*(pred[h] - true).T)
                fde[name][h].append(np.nanmean(err))
                if name in oof:
                    oof[name][h][idx_test] = pred[h]
            if name == "Floresta aleatória":
                importances.append(model.get_feature_importances())
    return fde, pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False), oof, folds


def fig_error(fde, path):
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    x = [1, 3, 5]
    ends = []
    for name, _, color, ls, marker in MODELS:
        y = [np.mean(fde[name][h]) for h in HORIZONS]
        ax.plot(x, y, color=color, linestyle=ls, marker=marker, markersize=4.5, linewidth=1.6)
        ends.append([y[-1], name, color])
    ends.sort()
    for i in range(1, len(ends)):  # afasta os rótulos que ficariam sobrepostos
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.75)
    for y, name, color in ends:
        ax.text(5.12, y, name, color=color if color != MUTED else INK, fontsize=7.5, va="center")
    style(ax)
    ax.set_xticks(x, ["1 s", "3 s", "5 s"])
    ax.set_xlim(0.8, 5.1)
    ax.set_ylim(0, None)
    ax.set_xlabel("Horizonte de previsão", fontsize=8.5, color=INK)
    ax.set_ylabel("Erro final (m)", fontsize=8.5, color=INK)
    fig.subplots_adjust(left=0.11, right=0.62, top=0.97, bottom=0.16)
    fig.savefig(path)
    plt.close(fig)


def fig_importances(imp, path):
    top = imp.head(6)[::-1]
    fig, ax = plt.subplots(figsize=(2.3, 1.75))
    bars = ax.barh([FEATURE_NAMES.get(k, k) for k in top.index], top.values, color=BLUE, height=0.62)
    for bar, value in zip(bars, top.values):
        ax.text(value + 0.006, bar.get_y() + bar.get_height() / 2, f"{value:.2f}".replace(".", ","), va="center", fontsize=7, color=INK)
    style(ax)
    ax.grid(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_xticks([])
    ax.tick_params(axis="y", labelsize=7, colors=INK)
    ax.set_xlim(0, top.max() * 1.22)
    fig.subplots_adjust(left=0.57, right=0.99, top=0.99, bottom=0.02)
    fig.savefig(path)
    plt.close(fig)


def classify(df, folds):
    out = {}
    configs = {
        "Floresta aleatória": {"model_type": "random_forest", "n_estimators": 60, "max_depth": 8},
        "Gradient boosting": {"model_type": "gradient_boosting", "max_iter": 80, "max_depth": 5},
    }
    for label, kwargs in configs.items():
        pred = np.empty(len(df), dtype=object)
        for idx_train, idx_test in folds:
            clf = ManeuverClassifier(**kwargs).fit(df.iloc[idx_train])
            pred[idx_test] = clf.predict(df.iloc[idx_test])
        out[label] = pred
    return out


def fig_confusion(true, pred, path):
    cm = confusion_matrix(true, pred, labels=MANEUVERS, normalize="true")
    fig, ax = plt.subplots(figsize=(2.35, 2.05))
    ax.imshow(cm, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#ffffff", "#104281"]), vmin=0, vmax=1)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{cm[i, j]:.2f}".replace(".", ","), ha="center", va="center", fontsize=6.5,
                    color="white" if cm[i, j] > 0.5 else INK)
    ax.set_xticks(range(5), MANEUVER_NAMES, rotation=35, ha="right", fontsize=7)
    ax.set_yticks(range(5), MANEUVER_NAMES, fontsize=7)
    ax.tick_params(length=0, colors=INK)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.set_xlabel("Manobra prevista", fontsize=7.5, color=INK)
    ax.set_ylabel("Manobra real", fontsize=7.5, color=INK)
    fig.subplots_adjust(left=0.30, right=0.99, top=0.99, bottom=0.30)
    fig.savefig(path)
    plt.close(fig)


def fig_example(df, oof, path):
    """Uma viragem real: escolhe-se o caso mediano entre os 5% em que a velocidade constante mais erra a 5 s."""
    true5 = df[["future_dx_5s", "future_dy_5s"]].to_numpy()
    err_cv = np.hypot(*(oof["Velocidade constante"]["5s"] - true5).T)
    turning = df.maneuver_class.isin(["TURN_LEFT", "TURN_RIGHT"]).to_numpy() & np.isfinite(err_cv)
    candidates = np.where(turning & (err_cv >= np.nanpercentile(err_cv[turning], 90)))[0]
    i = candidates[np.argsort(err_cv[candidates])[len(candidates) // 2]]
    row = df.iloc[i]
    cos_h, sin_h = np.cos(row.agent_heading), np.sin(row.agent_heading)

    def local(points):  # mapa -> referencial do veículo (frente, esquerda)
        p = np.asarray(points)
        return np.column_stack([cos_h * p[:, 0] + sin_h * p[:, 1], -sin_h * p[:, 0] + cos_h * p[:, 1]])

    series = [
        ("Trajetória real", [[row[f"future_dx_{h}"], row[f"future_dy_{h}"]] for h in HORIZONS], INK, "-", "o"),
        ("Velocidade constante", [oof["Velocidade constante"][h][i] for h in HORIZONS], MUTED, "--", "s"),
        ("Floresta aleatória", [oof["Floresta aleatória"][h][i] for h in HORIZONS], BLUE, "-", "D"),
    ]
    fig, ax = plt.subplots(figsize=(1.9, 2.05))
    for name, pts, color, ls, marker in series:
        xy = np.vstack([[0, 0], local(pts)])
        ax.plot(xy[:, 1], xy[:, 0], color=color, linestyle=ls, marker=marker, markersize=3.5, linewidth=1.4, label=name)
    real = local(series[0][1])
    for (fwd, left), h in zip(real, ["1 s", "3 s", "5 s"]):
        ax.annotate(h, (left, fwd), textcoords="offset points", xytext=(-7, 5), ha="right", fontsize=6.5, color=MUTED)
    ax.set_aspect("equal", adjustable="datalim")
    style(ax)
    ax.grid(color=GRID, linewidth=0.6)
    ax.invert_xaxis()  # esquerda do veículo à esquerda da figura
    ax.set_xlabel("Desvio lateral (m)", fontsize=7.5, color=INK)
    ax.set_ylabel("Avanço (m)", fontsize=7.5, color=INK)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.5, frameon=False, loc="lower left", labelcolor=INK, handlelength=1.6, borderaxespad=0.2)
    fig.subplots_adjust(left=0.22, right=0.97, top=0.98, bottom=0.19)
    fig.savefig(path)
    plt.close(fig)
    errors = {name: float(np.hypot(*(np.asarray(pts[-1]) - np.asarray(series[0][1][-1])))) for name, pts, *_ in series[1:]}
    return row.scenario_id, int(row.current_time_step), row.maneuver_class, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default="data/processed/predict_to_plan_samples.csv")
    parser.add_argument("--out", default="artigo_final/figuras")
    parser.add_argument("--splits", type=int, default=5)
    args = parser.parse_args()
    warnings.filterwarnings("ignore", message="Lead-vehicle columns are missing")

    df = pd.read_csv(args.csv).reset_index(drop=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Amostras: {len(df)} | cenas: {df.scenario_id.nunique()} | com alvo a 5 s: {df.future_dx_5s.notna().sum()}")

    fde, imp, oof, folds = out_of_fold(df, args.splits)
    print("\nErro final (m), média ± desvio-padrão das partes:")
    for name, *_ in MODELS:
        print(f"  {name:<28}" + "  ".join(f"{np.mean(fde[name][h]):6.2f} ± {np.std(fde[name][h]):.2f}" for h in HORIZONS))
    print("\nImportância das variáveis (floresta aleatória, média das partes e dos horizontes):")
    for key, value in imp.head(10).items():
        print(f"  {key:<22}{value:.3f}")

    fig_error(fde, out / "erro_horizonte.pdf")
    fig_importances(imp, out / "importancias.pdf")

    preds = classify(df, folds)
    true = df.maneuver_class.to_numpy()
    print("\nClassificação de manobras (fora da amostra):")
    print(f"  classe mais frequente: {pd.Series(true).value_counts(normalize=True).iloc[0] * 100:.1f}%")
    for label, pred in preds.items():
        print(f"  {label:<20} exatidão {accuracy_score(true, pred) * 100:.1f}% | F1 macro {f1_score(true, pred, average='macro'):.3f}")
    recall = confusion_matrix(true, preds["Floresta aleatória"], labels=MANEUVERS, normalize="true").diagonal()
    print("  acerto por classe (floresta aleatória):", dict(zip(MANEUVER_NAMES, recall.round(2))))
    fig_confusion(true, preds["Floresta aleatória"], out / "confusao.pdf")

    scenario, step, maneuver, errors = fig_example(df, oof, out / "exemplo.pdf")
    print(f"\nExemplo: cena {scenario}, instante {step}, manobra {maneuver}; erro a 5 s: "
          + ", ".join(f"{k} {v:.1f} m" for k, v in errors.items()))
    print(f"\nFiguras em {out}/")


if __name__ == "__main__":
    main()
