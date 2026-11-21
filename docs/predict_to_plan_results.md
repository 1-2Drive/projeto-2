# Resultados Experimentais: Prever para Planear (Predict to Plan)

## 1. Benchmark de Modelos de Predição de Trajetória (Regressão Contínua)

| model_name                          |   ade_1s |   fde_1s |   ade_3s |   fde_3s |   ade_5s |   fde_5s |   mean_inference_time_ms |
|:------------------------------------|---------:|---------:|---------:|---------:|---------:|---------:|-------------------------:|
| Constant Velocity (CV)              |    0.588 |    0.588 |    2.444 |    4.301 |    5.379 |   11.248 |                   0.0001 |
| Constant Acceleration (CA)          |    0.401 |    0.401 |    1.754 |    3.107 |    4.892 |   11.17  |                   0.0001 |
| Decision Tree (depth=6)             |    1.717 |    1.717 |    3.947 |    6.177 |    6.79  |   12.476 |                   0.0003 |
| Random Forest (trees=60)            |    0.821 |    0.821 |    2.304 |    3.786 |    4.697 |    9.485 |                   0.0084 |
| Gradient Boosting (HistGB, iter=80) |    0.48  |    0.48  |    1.766 |    3.052 |    4.112 |    8.806 |                   0.791  |

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

| Planning Modality            |   Scenarios Tested |   Collision Rate (%) |   Min Clearance (m) |   Emergency Decels / Run |   Avg Speed (m/s) |
|:-----------------------------|-------------------:|---------------------:|--------------------:|-------------------------:|------------------:|
| Uninformed (Reactive)        |                 30 |                 20   |                6.61 |                     0.3  |              6.23 |
| Constant Velocity            |                 30 |                 23.3 |                6.51 |                     0    |              6.29 |
| ML Predicted (Random Forest) |                 30 |                 16.7 |                6.58 |                     0    |              6.33 |
| Oracle (Ground Truth)        |                 30 |                 13.3 |                6.54 |                     0.07 |              6.3  |

![Resultados do Benchmark](../reports/figures/predict_to_plan_benchmark.png)
