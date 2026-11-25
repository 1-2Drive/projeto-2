# Resultados Experimentais: Prever para Planear (Predict to Plan)

## 1. Benchmark de Modelos de Predição de Trajetória (Regressão Contínua)

| model_name                           |   ade_1s |   fde_1s |   ade_3s |   fde_3s |   ade_5s |   fde_5s |   mean_inference_time_ms |   mae |   rmse |    r2 |   mae_3s |   rmse_3s |   r2_3s |
|:-------------------------------------|---------:|---------:|---------:|---------:|---------:|---------:|-------------------------:|------:|-------:|------:|---------:|----------:|--------:|
| Constant Velocity (CV)               |    0.588 |    0.588 |    2.444 |    4.301 |    5.379 |   11.248 |                   0.0002 | 3.263 |  6.197 | 0.943 |    2.636 |     3.813 |   0.958 |
| Constant Acceleration (CA)           |    0.401 |    0.401 |    1.754 |    3.107 |    4.892 |   11.17  |                   0.0001 | 3.024 |  6.298 | 0.947 |    1.935 |     2.982 |   0.975 |
| Decision Tree (depth=6)              |    1.717 |    1.717 |    3.947 |    6.177 |    6.79  |   12.476 |                   0.0002 | 4.142 |  7.053 | 0.905 |    3.81  |     5.386 |   0.917 |
| Random Forest (trees=60)             |    0.821 |    0.821 |    2.304 |    3.786 |    4.697 |    9.485 |                   0.0083 | 2.819 |  5.198 | 0.957 |    2.307 |     3.316 |   0.968 |
| Gradient Boosting (HistGB, iter=80)  |    0.48  |    0.48  |    1.766 |    3.052 |    4.112 |    8.806 |                   0.017  | 2.506 |  4.793 | 0.966 |    1.878 |     2.782 |   0.977 |
| Gradient Boosting (HistGB, iter=100) |    0.493 |    0.493 |    1.779 |    3.065 |    4.159 |    8.917 |                   0.0297 | 2.512 |  4.84  | 0.965 |    1.878 |     2.809 |   0.977 |

## 2. Benchmark de Classificação de Intenção / Manobra (Classificação Multiclasse)

| Classifier                                |   Accuracy |   F1-Macro |   F1-Weighted |
|:------------------------------------------|-----------:|-----------:|--------------:|
| Random Forest Classifier (trees=60)       |     0.6984 |     0.6712 |        0.7137 |
| HistGradientBoosting Classifier (iter=80) |     0.7763 |     0.732  |        0.7843 |

## 3. Importância das Features (Random Forest)

|                     |          0 |
|:--------------------|-----------:|
| agent_vy            | 0.465645   |
| agent_vx            | 0.41237    |
| agent_speed         | 0.053008   |
| agent_accel_est     | 0.0218307  |
| past_dy_1s          | 0.00960891 |
| past_dx_1s          | 0.00614921 |
| past_heading_change | 0.00590285 |
| yaw_rate_1s         | 0.00550793 |
| past_speed_mean     | 0.00515636 |
| ego_speed           | 0.00369828 |

## 4. Avaliação em Malha Fechada do Planeador A* com Evasão de Múltiplos Obstáculos

| Planning Modality           |   Scenarios Tested | OBB Collision Rate (%)   | Clearance < 3.5m (%)   |   Emergency Decels / Run |   Avg Speed (m/s) |   Max Lateral (m) |   Expansions |   Latency (ms) |
|:----------------------------|-------------------:|:-------------------------|:-----------------------|-------------------------:|------------------:|------------------:|-------------:|---------------:|
| Uninformed (1D Reactive)    |                 99 | 4.0% [1.6, 9.9]          | 26.3% [18.6, 35.7]     |                     0.21 |              6.83 |              0    |        806.9 |          26.15 |
| Constant Velocity (1D)      |                 99 | 5.1% [2.2, 11.3]         | 24.2% [16.9, 33.5]     |                     0.01 |              6.76 |              0    |        790   |          26.82 |
| ML Predicted (1D A*)        |                 99 | 5.1% [2.2, 11.3]         | 24.2% [16.9, 33.5]     |                     0.03 |              6.71 |              0    |        784.8 |          26.76 |
| ML Predicted (2D Frenet A*) |                 99 | 5.1% [2.2, 11.3]         | 26.3% [18.6, 35.7]     |                     0    |              6.59 |              0.02 |       1154.6 |         229.11 |
| Dijkstra Baseline (1D, h=0) |                 99 | 5.1% [2.2, 11.3]         | 27.3% [19.5, 36.8]     |                     0.04 |              6.56 |              0    |         81.9 |           2.78 |
| Greedy Baseline (1D, f=h)   |                 99 | 15.2% [9.4, 23.5]        | 35.4% [26.6, 45.2]     |                     0    |              8.52 |              0    |         31   |           1.03 |
| Oracle (1D Ground Truth)    |                 99 | 5.1% [2.2, 11.3]         | 18.2% [11.8, 26.9]     |                     0.03 |              6.68 |              0    |        784.9 |          26.21 |

![Resultados do Benchmark](../reports/figures/predict_to_plan_benchmark.png)
