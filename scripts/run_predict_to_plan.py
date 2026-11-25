#!/usr/bin/env python3
"""
Complete Predict-to-Plan (Prever para Planear) Evaluation Pipeline.
Combines:
1. Waymo Open Motion Dataset scenario sequence extraction
2. ML Trajectory Prediction (Physics Baselines vs Decision Tree vs Random Forest)
3. Spatiotemporal A* Speed and Trajectory Planning
4. Closed-loop safety and collision evaluation
5. Generation of metrics tables and scientific figures
"""

import os
import sys
from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure project root is in python path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.tfrecord_reader import discover_tfrecords, iter_scenarios
from src.preprocessing.sequence_extractor import SequenceExtractor
from src.features.pandas_feature_pipeline import PandasFeaturePipeline
from src.models.trajectory_predictor import (
    ConstantAccelerationPredictor,
    ConstantVelocityPredictor,
    ManeuverClassifier,
    MLTrajectoryPredictor,
    cross_validate_predictors,
    evaluate_maneuver_classifier,
    evaluate_trajectory_predictor,
    tune_trajectory_predictor,
)
from src.planning.astar_planner import (
    AStarTrajectoryPlanner,
    DijkstraTrajectoryPlanner,
    GreedyTrajectoryPlanner,
    compute_wilson_score_interval,
)


def main():
    print("=" * 70)
    print("🚀 WAYMO PREDICT-TO-PLAN (PREVER PARA PLANEAR) PIPELINE")
    print("=" * 70)

    # 1. Discover dataset shards (exclude testing shards since testing shards lack future ground truth)
    all_shards = discover_tfrecords()
    shards = [s for s in all_shards if not s.name.startswith("testing")]
    print(f"\n[1/5] Discovered {len(shards)} valid training/validation shards:")
    total_mb = sum(s.stat().st_size for s in shards) / (1024 * 1024)
    for s in shards:
        print(f"  - {s.name} ({s.stat().st_size / (1024*1024):.1f} MB)")
    print(f"Total dataset size available: {total_mb:.1f} MB")

    # 2. Extract or load interactive vehicle trajectory sequences
    processed_dir = ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    csv_path = processed_dir / "predict_to_plan_samples.csv"

    df_samples = None
    if csv_path.exists() and os.path.getsize(csv_path) > 10000:
        print(f"\n[2/5] Loading existing extracted samples from {csv_path}...")
        df_samples = pd.read_csv(csv_path)
        required_cols = set(PandasFeaturePipeline.FEATURE_COLS) | {"lead_gap", "lead_speed", "lead_closing_speed", "maneuver_class"}
        missing_cols = sorted(required_cols - set(df_samples.columns))
        lead_cols = {"lead_gap", "lead_speed", "lead_closing_speed"}
        if missing_cols and not shards and set(missing_cols) <= lead_cols:
            print("  Saved samples have no lead-vehicle features and there are no TFRecord files to extract them")
            print("  again: using the saved samples as they are (the models assume no vehicle ahead).")
        elif missing_cols:
            print(f"  Saved samples are from an older version (missing {missing_cols}): extracting again.")
            df_samples = None
        else:
            print(f"✓ Loaded {len(df_samples)} interactive obstacle samples.")
    if df_samples is None:
        print("\n[2/5] Extracting interactive obstacle sequences across scenarios...")
        extractor = SequenceExtractor(
            history_steps=10,
            future_steps=30,
            max_distance_to_ego=50.0,
            min_agent_speed=1.0,
            step_stride=10,
        )

        all_samples = []
        scenarios_parsed = 0
        max_scenarios_to_process = 60

        for scenario in iter_scenarios(shards, max_scenarios=max_scenarios_to_process):
            samples = extractor.extract_from_scenario(scenario)
            all_samples.extend(samples)
            scenarios_parsed += 1
            if scenarios_parsed % 10 == 0:
                print(f"  Parsed {scenarios_parsed} scenarios -> {len(all_samples)} trajectory samples...")

        df_samples = extractor.samples_to_dataframe(all_samples)
        # Same cleaning, feature and labelling steps as scripts/preprocess_at_scale.py
        df_samples = PandasFeaturePipeline.clean_and_filter_outliers(df_samples)
        df_samples = PandasFeaturePipeline.add_vectorized_kinematic_features(df_samples)
        df_samples = PandasFeaturePipeline.assign_maneuver_labels(df_samples)
        print(f"✓ Extraction complete: {len(df_samples)} interactive obstacle samples from {scenarios_parsed} scenarios.")
        df_samples.to_csv(csv_path, index=False)
        print(f"✓ Saved extracted samples to {csv_path}")

    # 3. Benchmarking Trajectory Predictors & Maneuver Classifiers
    print("\n[3/5] Benchmarking Trajectory Predictors & Maneuver Classifiers (Physics vs ML)...")
    df_train, df_test = PandasFeaturePipeline.create_scenario_based_split(df_samples, test_size=0.25, random_state=42)

    print(f"  Train set: {len(df_train)} samples across {df_train['scenario_id'].nunique()} scenarios")
    print(f"  Test set:  {len(df_test)} samples across {df_test['scenario_id'].nunique()} scenarios")

    # "[global frame]" = first version of the ML predictor (map coordinates, predicts the displacement itself).
    # The others work in the vehicle's own frame and predict the correction to constant velocity.
    predictors = [
        ConstantVelocityPredictor(),
        ConstantAccelerationPredictor(),
        MLTrajectoryPredictor(model_type="random_forest", frame="global", residual=False, n_estimators=60, max_depth=8),
        MLTrajectoryPredictor(model_type="decision_tree", max_depth=6),
        MLTrajectoryPredictor(model_type="random_forest", n_estimators=60, max_depth=8),
        MLTrajectoryPredictor(model_type="gradient_boosting", max_iter=80, max_depth=5),
    ]

    print("\n  🔍 Tuning HistGradientBoosting Regressor via RandomizedSearchCV...")
    tuned_gb, best_params_gb = tune_trajectory_predictor(
        df_train, model_type="gradient_boosting", n_iter=10, cv=3, random_state=42
    )
    print(f"    ✓ Best Hyperparameters: {best_params_gb}")
    predictors.append(tuned_gb)

    metrics_list = []
    trained_models = {}

    for p in predictors:
        if isinstance(p, MLTrajectoryPredictor) and p != tuned_gb:
            print(f"  Training {p.name}...")
            p.fit(df_train)
        if isinstance(p, MLTrajectoryPredictor) and p != tuned_gb:
            print(f"  Training {p.name}...")
            p.fit(df_train)
            if getattr(p, "frame", "agent") == "agent":
                trained_models[p.model_type] = p
        elif p == tuned_gb:
            trained_models["tuned_gradient_boosting"] = tuned_gb

        metrics, preds = evaluate_trajectory_predictor(p, df_test)
        metrics_list.append(metrics)
        print(
            f"    -> {metrics.model_name}: "
            f"ADE@3s = {metrics.ade_3s:.3f}m | FDE@3s = {metrics.fde_3s:.3f}m | "
            f"MAE = {metrics.mae:.3f}m | RMSE = {metrics.rmse:.3f}m | R² = {metrics.r2:.3f} | "
            f"Latency = {metrics.mean_inference_time_ms:.3f}ms"
        )

    df_metrics = pd.DataFrame([m.__dict__ for m in metrics_list])
    print("\n--- ML Trajectory Prediction Benchmark Results (Standard & Regression Metrics) ---")
    cols_to_show = ["model_name", "ade_1s", "fde_1s", "ade_3s", "fde_3s", "mae", "rmse", "r2", "mean_inference_time_ms"]
    print(df_metrics[cols_to_show].to_string(index=False))

    # Scenario-grouped cross-validation: each scenario is tested once, never seen in training
    print("\n--- 5-fold cross-validation grouped by scenario (final error in metres, mean ± std) ---")
    df_cv = cross_validate_predictors(
        {
            "Constant Velocity (CV)": lambda: ConstantVelocityPredictor(),
            "Constant Acceleration (CA)": lambda: ConstantAccelerationPredictor(),
            "Random Forest [global frame]": lambda: MLTrajectoryPredictor(
                model_type="random_forest", frame="global", residual=False, n_estimators=60, max_depth=8
            ),
            "Decision Tree": lambda: MLTrajectoryPredictor(model_type="decision_tree", max_depth=6),
            "Random Forest": lambda: MLTrajectoryPredictor(model_type="random_forest", n_estimators=60, max_depth=8),
            "Gradient Boosting": lambda: MLTrajectoryPredictor(model_type="gradient_boosting", max_iter=80, max_depth=5),
        },
        df_samples,
        n_splits=5,
    )
    print(df_cv.to_string(index=False))

    # Feature Importance analysis
    rf_model = trained_models.get("random_forest")
    importances = None
    if rf_model:
        importances = rf_model.get_feature_importances()
        print("\n--- Top Predictive Features (Random Forest) ---")
        print(importances.head(8).to_string())

    # Maneuver Intent Classification Benchmark
    print("\n--- Maneuver Intent Classification Benchmark ---")
    classifiers = [
        ManeuverClassifier(model_type="random_forest", frame="global", n_estimators=60, max_depth=8),
        ManeuverClassifier(model_type="random_forest", n_estimators=60, max_depth=8),
        ManeuverClassifier(model_type="gradient_boosting", max_iter=80, max_depth=5),
    ]
    clf_metrics = []
    for clf in classifiers:
        print(f"  Training {clf.name}...")
        clf.fit(df_train)
        res = evaluate_maneuver_classifier(clf, df_test)
        clf_metrics.append({
            "Classifier": clf.name,
            "Accuracy": res["accuracy"],
            "F1-Macro": res["f1_macro"],
            "F1-Weighted": res["f1_weighted"],
        })
        print(f"    -> {clf.name}: Accuracy = {res['accuracy']:.4f} | F1-Macro = {res['f1_macro']:.4f} | F1-Weighted = {res['f1_weighted']:.4f}")

    df_clf_metrics = pd.DataFrame(clf_metrics)

    # 4. Multi-Obstacle Closed-Loop Planning Evaluation
    print("\n[4/5] Running Spatiotemporal A* Speed & Collision Planner (Multi-Obstacle)...")
    planner_1d = AStarTrajectoryPlanner(
        horizon_steps=30,  # 3.0 seconds horizon
        dt=0.1,
        max_speed=18.0,
        safety_radius=3.5,
    )
    planner_2d = AStarTrajectoryPlanner(
        horizon_steps=30,
        dt=0.1,
        max_speed=18.0,
        safety_radius=3.5,
        max_expansions=3000,
    )
    planner_dijkstra = DijkstraTrajectoryPlanner(
        horizon_steps=30,
        dt=0.1,
        max_speed=18.0,
        safety_radius=3.5,
    )
    planner_greedy = GreedyTrajectoryPlanner(
        horizon_steps=30,
        dt=0.1,
        max_speed=18.0,
        safety_radius=3.5,
    )

    # Evaluate closed-loop planning across test scenarios
    plan_modes = [
        "Uninformed (1D Reactive)",
        "Constant Velocity (1D)",
        "ML Predicted (1D A*)",
        "ML Predicted (2D Frenet A*)",
        "Dijkstra Baseline (1D, h=0)",
        "Greedy Baseline (1D, f=h)",
        "Oracle (1D Ground Truth)",
    ]
    planning_results = {mode: [] for mode in plan_modes}

    # Group test samples by (scenario_id, current_time_step) to find multi-obstacle scenes
    scene_groups = df_test.groupby(["scenario_id", "current_time_step"])
    valid_scenes = []
    for (scen_id, t_step), group in scene_groups:
        # Check if there is at least one interactive obstacle ahead of ego
        has_critical = (
            (group["distance_to_ego"] < 25.0) &
            (group["rel_y"].abs() < 4.0) &
            (group["rel_x"] > 1.0)
        ).any()
        if has_critical:
            valid_scenes.append(group)

    print(f"  Found {len(valid_scenes)} multi-obstacle safety-critical interactive test scenes.")
    eval_scenes = valid_scenes  # Evaluate all valid test scenes
    total_obstacles_evaluated = sum(len(g) for g in eval_scenes)
    print(f"  Evaluating {len(eval_scenes)} distinct scenes with a total of {total_obstacles_evaluated} simultaneous dynamic obstacles...")

    for scene_df in eval_scenes:
        ego_sample = scene_df.iloc[0]
        ego_v = float(ego_sample["ego_speed"])
        ego_heading = float(ego_sample["ego_heading"])

        # Longitudinal reference path along ego heading
        path_length = 60.0
        s_pts = np.linspace(0, path_length, 60)
        ego_path = np.column_stack([
            ego_sample["ego_x"] + s_pts * np.cos(ego_heading),
            ego_sample["ego_y"] + s_pts * np.sin(ego_heading),
        ])

        gt_obstacles = {}
        uninformed_obs = {}
        cv_obs = {}
        ml_obs = {}
        gt_headings = {}

        for _, obs_row in scene_df.iterrows():
            tid = int(obs_row["track_id"])
            ox0 = float(obs_row["agent_x"])
            oy0 = float(obs_row["agent_y"])
            ovx = float(obs_row["agent_vx"])
            ovy = float(obs_row["agent_vy"])
            o_head = float(obs_row["agent_heading"])
            gt_headings[tid] = o_head

            # 1) Ground truth trajectory with piecewise linear interpolation (1s and 3s)
            p1_x = ox0 + float(obs_row["future_dx_1s"])
            p1_y = oy0 + float(obs_row["future_dy_1s"])
            p3_x = ox0 + float(obs_row["future_dx_3s"])
            p3_y = oy0 + float(obs_row["future_dy_3s"])
            gt_traj = []
            for step in range(31):
                if step <= 10:
                    ratio = step / 10.0
                    gt_traj.append((ox0 + (p1_x - ox0) * ratio, oy0 + (p1_y - oy0) * ratio))
                else:
                    ratio = (step - 10) / 20.0
                    gt_traj.append((p1_x + (p3_x - p1_x) * ratio, p1_y + (p3_y - p1_y) * ratio))
            gt_obstacles[tid] = gt_traj

            # 2) Uninformed trajectory: obstacle assumed stationary
            uninformed_obs[tid] = [(ox0, oy0)] * 31

            # 3) Constant Velocity trajectory
            cv_obs[tid] = [(ox0 + ovx * (step * 0.1), oy0 + ovy * (step * 0.1)) for step in range(31)]

            # 4) ML Predicted trajectory (piecewise 1s and 3s predictions)
            active_ml = trained_models.get("tuned_gradient_boosting") or rf_model
            if active_ml is not None:
                ml_p = active_ml.predict_displacements(pd.DataFrame([obs_row]))
                m1_x = ox0 + float(ml_p["1s"][0, 0])
                m1_y = oy0 + float(ml_p["1s"][0, 1])
                m3_x = ox0 + float(ml_p["3s"][0, 0])
                m3_y = oy0 + float(ml_p["3s"][0, 1])
                ml_traj = []
                for step in range(31):
                    if step <= 10:
                        ratio = step / 10.0
                        ml_traj.append((ox0 + (m1_x - ox0) * ratio, oy0 + (m1_y - oy0) * ratio))
                    else:
                        ratio = (step - 10) / 20.0
                        ml_traj.append((m1_x + (m3_x - m1_x) * ratio, m1_y + (m3_y - m1_y) * ratio))
                ml_obs[tid] = ml_traj
            else:
                ml_obs[tid] = cv_obs[tid]

        # 1D A* configurations
        configs_1d = [
            ("Uninformed (1D Reactive)", planner_1d, uninformed_obs),
            ("Constant Velocity (1D)", planner_1d, cv_obs),
            ("ML Predicted (1D A*)", planner_1d, ml_obs),
            ("Dijkstra Baseline (1D, h=0)", planner_dijkstra, ml_obs),
            ("Greedy Baseline (1D, f=h)", planner_greedy, ml_obs),
            ("Oracle (1D Ground Truth)", planner_1d, gt_obstacles),
        ]

        for mode_name, pl, pred_obs in configs_1d:
            s_tr, v_tr, a_tr = pl.plan_trajectory(
                ego_init_v=ego_v,
                ego_path_xy=ego_path,
                predicted_obstacles=pred_obs,
            )
            res = pl.evaluate_plan_against_ground_truth(
                mode=mode_name,
                ego_path_xy=ego_path,
                s_traj=s_tr,
                v_traj=v_tr,
                a_traj=a_tr,
                gt_obstacles=gt_obstacles,
                ego_heading=ego_heading,
                gt_headings=gt_headings,
                obb_safety_margin=0.1,
            )
            planning_results[mode_name].append(res)

        # 2D Frenet Spatiotemporal A* with Lateral Evasion
        s_2d, v_2d, a_2d, d_2d = planner_2d.plan_trajectory_2d(
            ego_init_v=ego_v,
            ego_path_xy=ego_path,
            predicted_obstacles=ml_obs,
            lat_options=[-1.5, 0.0, 1.5],
            gt_headings=gt_headings,
            use_obb=True,
        )
        res_2d = planner_2d.evaluate_plan_against_ground_truth(
            mode="ML Predicted (2D Frenet A*)",
            ego_path_xy=ego_path,
            s_traj=s_2d,
            v_traj=v_2d,
            a_traj=a_2d,
            gt_obstacles=gt_obstacles,
            ego_heading=ego_heading,
            gt_headings=gt_headings,
            obb_safety_margin=0.1,
            d_traj=d_2d,
        )
        planning_results["ML Predicted (2D Frenet A*)"].append(res_2d)

    # Summarize planning results
    plan_summary_rows = []
    for mode_name, results in planning_results.items():
        if not results:
            continue
        total_runs = len(results)
        obb_col_count = sum(1 for r in results if r.obb_collision_occurred)
        obb_rate, obb_lower, obb_upper = compute_wilson_score_interval(obb_col_count, total_runs)

        circle_col_count = sum(1 for r in results if r.collision_occurred)
        circle_rate, circ_lower, circ_upper = compute_wilson_score_interval(circle_col_count, total_runs)

        avg_min_dist = float(np.mean([r.min_distance_to_obstacle for r in results]))
        avg_emergency = float(np.mean([r.emergency_braking_events for r in results]))
        avg_speed = float(np.mean([r.average_speed for r in results]))
        avg_lat = float(np.mean([r.max_lateral_displacement for r in results]))
        avg_exp = float(np.mean([r.expansions for r in results]))
        avg_lat_ms = float(np.mean([r.planning_time_ms for r in results]))

        plan_summary_rows.append({
            "Planning Modality": mode_name,
            "Scenarios Tested": total_runs,
            "OBB Collision Rate (%)": f"{obb_rate:.1f}% [{obb_lower:.1f}, {obb_upper:.1f}]",
            "Clearance < 3.5m (%)": f"{circle_rate:.1f}% [{circ_lower:.1f}, {circ_upper:.1f}]",
            "Emergency Decels / Run": round(avg_emergency, 2),
            "Avg Speed (m/s)": round(avg_speed, 2),
            "Max Lateral (m)": round(avg_lat, 2),
            "Expansions": round(avg_exp, 1),
            "Latency (ms)": round(avg_lat_ms, 2),
        })

    df_plan_summary = pd.DataFrame(plan_summary_rows)
    print("\n--- Spatiotemporal Planning & Collision Benchmark (Full Test Set) ---")
    print(df_plan_summary.to_string(index=False))

    # 5. Generate Figures and Artifacts
    print("\n[5/5] Generating Evaluation Figures and Final Report...")
    figures_dir = ROOT / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: ML Model ADE/FDE comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x_indices = np.arange(len(df_metrics))
    width = 0.35

    axes[0].bar(x_indices - width/2, df_metrics["ade_1s"], width, label="ADE @ 1.0s", color="#3b82f6")
    axes[0].bar(x_indices + width/2, df_metrics["ade_3s"], width, label="ADE @ 3.0s", color="#1d4ed8")
    axes[0].set_ylabel("Displacement Error (meters)")
    axes[0].set_title("Average Displacement Error (ADE)")
    axes[0].set_xticks(x_indices)
    axes[0].set_xticklabels([m.split("(")[0].strip() for m in df_metrics["model_name"]], rotation=15)
    axes[0].legend()
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    # Collision rate comparison bar chart (OBB vs 3.5m Clearance)
    x_plan = np.arange(len(df_plan_summary))
    obb_vals = [float(row["OBB Collision Rate (%)"].split("%")[0]) for _, row in df_plan_summary.iterrows()]
    circ_vals = [float(row["Clearance < 3.5m (%)"].split("%")[0]) for _, row in df_plan_summary.iterrows()]
    axes[1].bar(x_plan - width/2, obb_vals, width, label="OBB Collision Rate (%)", color="#ef4444")
    axes[1].bar(x_plan + width/2, circ_vals, width, label="Clearance < 3.5m (%)", color="#f97316")
    axes[1].set_ylabel("Rate (%)")
    axes[1].set_title("Planner Safety: Real Bounding Box (OBB) vs 3.5m Buffer")
    axes[1].set_xticks(x_plan)
    axes[1].set_xticklabels([m.split("(")[0].strip() for m in df_plan_summary["Planning Modality"]], rotation=15)
    axes[1].legend()
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    chart_path = figures_dir / "predict_to_plan_benchmark.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"✓ Saved benchmark figures to {chart_path}")

    # Save summary tables to markdown
    docs_dir = ROOT / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    results_md_path = docs_dir / "predict_to_plan_results.md"
    with open(results_md_path, "w") as f:
        f.write("# Resultados Experimentais: Prever para Planear (Predict to Plan)\n\n")
        f.write("## 1. Benchmark de Modelos de Predição de Trajetória (Regressão Contínua)\n\n")
        f.write(df_metrics.to_markdown(index=False) + "\n\n")
        f.write("### Validação cruzada por cenário (5 partes; erro final em metros, média ± desvio-padrão)\n\n")
        f.write(df_cv.to_markdown(index=False) + "\n\n")
        f.write("## 2. Benchmark de Classificação de Intenção / Manobra (Classificação Multiclasse)\n\n")
        f.write(df_clf_metrics.to_markdown(index=False) + "\n\n")
        if rf_model and importances is not None:
            f.write("## 3. Importância das Features (Random Forest)\n\n")
            f.write(importances.head(10).to_markdown() + "\n\n")
        f.write("## 4. Avaliação em Malha Fechada do Planeador A* com Evasão de Múltiplos Obstáculos\n\n")
        f.write(df_plan_summary.to_markdown(index=False) + "\n\n")
        f.write(f"![Resultados do Benchmark](../reports/figures/predict_to_plan_benchmark.png)\n")

    print(f"✓ Saved markdown results to {results_md_path}")
    print("\n✅ PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
