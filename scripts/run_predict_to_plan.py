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
from src.models.trajectory_predictor import (
    ConstantVelocityPredictor,
    ConstantAccelerationPredictor,
    MLTrajectoryPredictor,
    evaluate_trajectory_predictor,
)
from src.planning.astar_planner import AStarTrajectoryPlanner


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

    if csv_path.exists() and os.path.getsize(csv_path) > 10000:
        print(f"\n[2/5] Loading existing extracted samples from {csv_path}...")
        df_samples = pd.read_csv(csv_path)
        print(f"✓ Loaded {len(df_samples)} interactive obstacle samples.")
    else:
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
        print(f"✓ Extraction complete: {len(df_samples)} interactive obstacle samples from {scenarios_parsed} scenarios.")
        df_samples.to_csv(csv_path, index=False)
        print(f"✓ Saved extracted samples to {csv_path}")

    # 3. Benchmarking Trajectory Predictors (Physics vs ML)
    print("\n[3/5] Benchmarking Trajectory Predictors (Physics vs ML)...")
    unique_scenarios = list(df_samples["scenario_id"].unique())
    import random
    random.seed(42)
    random.shuffle(unique_scenarios)
    split_idx = int(len(unique_scenarios) * 0.75)
    train_scenarios = set(unique_scenarios[:split_idx])

    train_mask = df_samples["scenario_id"].isin(train_scenarios)
    df_train = df_samples[train_mask].copy()
    df_test = df_samples[~train_mask].copy()

    print(f"  Train set: {len(df_train)} samples across {len(train_scenarios)} scenarios")
    print(f"  Test set:  {len(df_test)} samples across {len(unique_scenarios) - len(train_scenarios)} scenarios")

    predictors = [
        ConstantVelocityPredictor(),
        ConstantAccelerationPredictor(),
        MLTrajectoryPredictor(model_type="decision_tree", max_depth=6),
        MLTrajectoryPredictor(model_type="random_forest", n_estimators=60, max_depth=8),
        MLTrajectoryPredictor(model_type="gradient_boosting", max_iter=80, max_depth=5),
    ]

    metrics_list = []
    trained_models = {}

    for p in predictors:
        if isinstance(p, MLTrajectoryPredictor):
            print(f"  Training {p.name}...")
            p.fit(df_train)
            trained_models[p.model_type] = p

        metrics, preds = evaluate_trajectory_predictor(p, df_test)
        metrics_list.append(metrics)
        print(f"    -> {metrics.model_name}: ADE@3s = {metrics.ade_3s:.3f}m | FDE@3s = {metrics.fde_3s:.3f}m | Latency = {metrics.mean_inference_time_ms:.3f}ms")

    df_metrics = pd.DataFrame([m.__dict__ for m in metrics_list])
    print("\n--- ML Trajectory Prediction Benchmark Results ---")
    print(df_metrics.to_string(index=False))

    # Feature Importance analysis
    rf_model = trained_models.get("random_forest")
    if rf_model:
        importances = rf_model.get_feature_importances()
        print("\n--- Top Predictive Features (Random Forest) ---")
        print(importances.head(8).to_string())

    # 4. Closed-Loop Planning Evaluation
    print("\n[4/5] Running Spatiotemporal A* Speed & Collision Planner...")
    planner = AStarTrajectoryPlanner(
        horizon_steps=30,  # 3.0 seconds horizon
        dt=0.1,
        max_speed=18.0,
        safety_radius=3.5,
    )

    # Evaluate closed-loop planning across test scenarios
    plan_modes = ["Uninformed (Reactive)", "Constant Velocity", "ML Predicted (Random Forest)", "Oracle (Ground Truth)"]
    planning_results = {mode: [] for mode in plan_modes}

    # Select representative close interactive encounters from test set (obstacle in front of ego)
    interactive_candidates = df_test[
        (df_test["distance_to_ego"] < 25.0) &
        (df_test["rel_y"].abs() < 4.0) &
        (df_test["rel_x"] > 2.0)
    ]
    test_samples = interactive_candidates.head(25)
    print(f"  Evaluating {len(test_samples)} safety-critical interactive scenarios...")

    for _, sample in test_samples.iterrows():
        ego_v = sample["ego_speed"]
        # Ego path straight ahead along ego heading
        path_length = 60.0
        s_pts = np.linspace(0, path_length, 60)
        ego_path = np.column_stack([
            sample["ego_x"] + s_pts * np.cos(sample["ego_heading"]),
            sample["ego_y"] + s_pts * np.sin(sample["ego_heading"]),
        ])

        # True obstacle trajectory over 30 steps
        obs_x0 = sample["agent_x"]
        obs_y0 = sample["agent_y"]
        obs_vx = sample["agent_vx"]
        obs_vy = sample["agent_vy"]

        # 1) Ground truth trajectory (using actual displacement)
        gt_traj = []
        for step in range(31):
            ratio = step / 30.0
            gx = obs_x0 + sample["future_dx_3s"] * ratio
            gy = obs_y0 + sample["future_dy_3s"] * ratio
            gt_traj.append((gx, gy))
        gt_obstacles = {sample["track_id"]: gt_traj}

        # 2) Uninformed: obstacle stays at x0, y0
        uninformed_traj = [(obs_x0, obs_y0)] * 31
        uninformed_obs = {sample["track_id"]: uninformed_traj}

        # 3) Constant Velocity: obs_x0 + vx * t
        cv_traj = [(obs_x0 + obs_vx * (step * 0.1), obs_y0 + obs_vy * (step * 0.1)) for step in range(31)]
        cv_obs = {sample["track_id"]: cv_traj}

        # 4) ML Predicted: using Random Forest
        rf_preds = rf_model.predict_displacements(pd.DataFrame([sample]))
        rf_dx3s = rf_preds["3s"][0, 0]
        rf_dy3s = rf_preds["3s"][0, 1]
        ml_traj = [(obs_x0 + rf_dx3s * (step / 30.0), obs_y0 + rf_dy3s * (step / 30.0)) for step in range(31)]
        ml_obs = {sample["track_id"]: ml_traj}

        # Plan for each modality
        configs = [
            ("Uninformed (Reactive)", uninformed_obs),
            ("Constant Velocity", cv_obs),
            ("ML Predicted (Random Forest)", ml_obs),
            ("Oracle (Ground Truth)", gt_obstacles),
        ]

        for mode_name, pred_obs in configs:
            s_tr, v_tr, a_tr = planner.plan_trajectory(
                ego_init_v=ego_v,
                ego_path_xy=ego_path,
                predicted_obstacles=pred_obs,
            )
            res = planner.evaluate_plan_against_ground_truth(
                mode=mode_name,
                ego_path_xy=ego_path,
                s_traj=s_tr,
                v_traj=v_tr,
                a_traj=a_tr,
                gt_obstacles=gt_obstacles,
            )
            planning_results[mode_name].append(res)

    # Summarize planning results
    plan_summary_rows = []
    for mode_name, results in planning_results.items():
        if not results:
            continue
        total_runs = len(results)
        collision_count = sum(1 for r in results if r.collision_occurred)
        collision_rate = (collision_count / total_runs) * 100.0
        avg_min_dist = float(np.mean([r.min_distance_to_obstacle for r in results]))
        avg_emergency = float(np.mean([r.emergency_braking_events for r in results]))
        avg_speed = float(np.mean([r.average_speed for r in results]))

        plan_summary_rows.append({
            "Planning Modality": mode_name,
            "Scenarios Tested": total_runs,
            "Collision Rate (%)": round(collision_rate, 1),
            "Min Clearance (m)": round(avg_min_dist, 2),
            "Emergency Decels / Run": round(avg_emergency, 2),
            "Avg Speed (m/s)": round(avg_speed, 2),
        })

    df_plan_summary = pd.DataFrame(plan_summary_rows)
    print("\n--- Spatiotemporal Planning & Collision Benchmark ---")
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

    # Collision rate comparison bar chart
    axes[1].bar(df_plan_summary["Planning Modality"], df_plan_summary["Collision Rate (%)"], color=["#ef4444", "#f97316", "#10b981", "#059669"])
    axes[1].set_ylabel("Collision Rate (%)")
    axes[1].set_title("Closed-Loop Planner Collision Rate Comparison")
    axes[1].set_xticklabels([m.split("(")[0].strip() for m in df_plan_summary["Planning Modality"]], rotation=15)
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
        f.write("## 1. Benchmark de Modelos de Predição de Trajetória (ML)\n\n")
        f.write(df_metrics.to_markdown(index=False) + "\n\n")
        if rf_model and importances is not None:
            f.write("## 2. Importância das Features (Random Forest)\n\n")
            f.write(importances.head(10).to_markdown() + "\n\n")
        f.write("## 3. Avaliação em Malha Fechada do Planeador A* com Evasão de Obstáculos\n\n")
        f.write(df_plan_summary.to_markdown(index=False) + "\n\n")
        f.write(f"![Resultados do Benchmark](../reports/figures/predict_to_plan_benchmark.png)\n")

    print(f"✓ Saved markdown results to {results_md_path}")
    print("\n✅ PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
