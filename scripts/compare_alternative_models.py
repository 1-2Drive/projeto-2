"""
Modelos alternativos para a previsão de trajetórias e curva de aprendizagem.

Compara, com a mesma representação (referencial do veículo + correção à
velocidade constante) e a mesma validação cruzada por cenário, modelos que não
fazem parte do preditor principal: regressão linear (ridge), k-vizinhos mais
próximos (com e sem normalização) e SVM de regressão. Serve para justificar no
artigo a escolha das árvores.

A segunda parte treina a floresta aleatória com cada vez mais cenas, para ver
se mais dados do mesmo tipo ainda ajudam.

Uso:
    python scripts/compare_alternative_models.py
    python scripts/compare_alternative_models.py --csv data/processed/predict_to_plan_samples.csv --skip-svr
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.trajectory_predictor import (
    MLTrajectoryPredictor,
    cross_validate_predictors,
    evaluate_trajectory_predictor,
)


def with_model(label, builder):
    """Um MLTrajectoryPredictor igual ao da floresta aleatória, mas com outro modelo por horizonte."""

    class Alternative(MLTrajectoryPredictor):
        def __init__(self):
            super().__init__("random_forest")
            self.name = label

        def _new_model(self):
            return builder()

    return Alternative


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default="data/processed/predict_to_plan_samples.csv")
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--skip-svr", action="store_true", help="salta o SVM, que é o mais lento")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    df = pd.read_csv(args.csv)
    print(f"Amostras: {len(df)} | cenas: {df.scenario_id.nunique()}\n")

    factories = {
        "Regressão linear (ridge), normalizada": with_model("ridge", lambda: make_pipeline(StandardScaler(), Ridge(1.0))),
        "k-vizinhos (k=15), sem normalizar": with_model("knn_raw", lambda: KNeighborsRegressor(15)),
        "k-vizinhos (k=15), normalizado": with_model("knn", lambda: make_pipeline(StandardScaler(), KNeighborsRegressor(15))),
        "Floresta aleatória (60 árvores, prof. 8)": lambda: MLTrajectoryPredictor(
            model_type="random_forest", n_estimators=60, max_depth=8
        ),
    }
    if not args.skip_svr:
        factories["SVM de regressão (RBF), normalizado"] = with_model(
            "svr", lambda: make_pipeline(StandardScaler(), MultiOutputRegressor(SVR(C=3.0, epsilon=0.1)))
        )

    print(f"{'modelo':<42}{'1 s':>14}{'3 s':>14}{'5 s':>15}   tempo")
    for label, factory in factories.items():
        t0 = time.time()
        row = cross_validate_predictors({label: factory}, df, args.splits).iloc[0]
        print(f"{label:<42}{row.fde_1s:>14}{row.fde_3s:>14}{row.fde_5s:>15}   {time.time() - t0:.0f} s", flush=True)

    print("\nCurva de aprendizagem (floresta aleatória, erro final a 3 s):")
    groups = df.scenario_id.to_numpy()
    rng = np.random.default_rng(0)
    folds = list(GroupKFold(n_splits=args.splits).split(df, groups=groups))
    max_train = min(len(pd.unique(groups[idx_train])) for idx_train, _ in folds)
    for n in sorted({10, 25, 50, 80, max_train}):
        if n > max_train:
            continue
        errors = []
        for idx_train, idx_test in folds:
            scenes = pd.unique(groups[idx_train])
            keep = set(rng.choice(scenes, size=n, replace=False))
            train = df.iloc[idx_train]
            model = MLTrajectoryPredictor(model_type="random_forest", n_estimators=60, max_depth=8)
            model.fit(train[train.scenario_id.isin(keep)])
            errors.append(evaluate_trajectory_predictor(model, df.iloc[idx_test])[0].fde_3s)
        print(f"  {n:>4} cenas de treino: {np.mean(errors):.2f} m")


if __name__ == "__main__":
    main()
