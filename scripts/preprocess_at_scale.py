#!/usr/bin/env python3
"""
Scalable Extraction Script:
Processes all downloaded training and validation TFRecord shards,
runs vectorized cleaning and feature engineering with NumPy and Pandas,
and writes out processed datasets and summary metadata.
"""

import sys
from pathlib import Path
import time
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.tfrecord_reader import discover_tfrecords, iter_scenarios
from src.preprocessing.sequence_extractor import SequenceExtractor
from src.features.pandas_feature_pipeline import PandasFeaturePipeline


def main():
    print("=" * 70)
    print("⚡ EXTRAÇÃO E PRÉ-PROCESSAMENTO VETORIZADO EM LARGA ESCALA (NUMPY & PANDAS)")
    print("=" * 70)

    all_shards = discover_tfrecords()
    train_shards = [s for s in all_shards if "training" in s.name or "validation" in s.name]
    print(f"\n[1/4] Shards identificados para processamento: {len(train_shards)}")
    for s in train_shards:
        print(f"  • {s.name} ({s.stat().st_size / (1024*1024):.1f} MB)")

    # 2. Sequence extraction
    print("\n[2/4] A extrair sequências temporais de obstáculos interativos...")
    extractor = SequenceExtractor(
        history_steps=10,
        future_steps=30,
        max_distance_to_ego=55.0,
        min_agent_speed=0.5,
        step_stride=10,
    )

    t0 = time.time()
    all_samples = []
    scenarios_count = 0
    max_scenarios = 180  # Scale across shards

    for scenario in iter_scenarios(train_shards, max_scenarios=max_scenarios):
        samples = extractor.extract_from_scenario(scenario)
        all_samples.extend(samples)
        scenarios_count += 1
        if scenarios_count % 20 == 0:
            print(f"  -> {scenarios_count} cenários processados | {len(all_samples)} amostras acumuladas...")

    raw_df = extractor.samples_to_dataframe(all_samples)
    print(f"✓ Extração inicial concluída: {len(raw_df)} registos em {time.time() - t0:.1f}s")

    # 3. Vectorized Preprocessing with NumPy & Pandas
    print("\n[3/4] A executar pipeline de pré-processamento e engenharia com Pandas & NumPy...")
    print(f"  - Registos antes da limpeza: {len(raw_df)}")

    # Clean & filter bounds
    clean_df = PandasFeaturePipeline.clean_and_filter_outliers(raw_df)
    print(f"  - Registos após filtragem física vetorial: {len(clean_df)} (retidos {len(clean_df)/max(1,len(raw_df))*100:.1f}%)")

    # Vectorized kinematic features (TTC, closing speed, bearing angle, accel)
    feat_df = PandasFeaturePipeline.add_vectorized_kinematic_features(clean_df)
    print(f"  - Features cinemáticas e de interação adicionadas: {list(feat_df.columns[-6:])}")

    # Maneuver classification
    final_df = PandasFeaturePipeline.assign_maneuver_labels(feat_df)
    maneuver_counts = final_df["maneuver_class"].value_counts()
    print("\n  - Distribuição das classes de manobras:")
    for cls, cnt in maneuver_counts.items():
        print(f"     * {cls:15}: {cnt:5d} ({cnt/len(final_df)*100:.1f}%)")

    # Compute summary stats table with pandas
    stats_df = PandasFeaturePipeline.compute_summary_statistics(final_df)
    print("\n--- Estatísticas Descritivas (Pandas Aggregation) ---")
    print(stats_df.to_string())

    # 4. Save processed artifacts
    processed_dir = ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_csv = processed_dir / "predict_to_plan_samples.csv"
    final_df.to_csv(out_csv, index=False)
    print(f"\n[4/4] Dataset final guardado: {out_csv} ({len(final_df)} linhas, {len(final_df.columns)} colunas)")

    stats_csv = processed_dir / "dataset_summary_statistics.csv"
    stats_df.to_csv(stats_csv)
    print(f"✓ Resumo estatístico guardado: {stats_csv}")


if __name__ == "__main__":
    main()
