# Resultados Experimentais: Prever para Planear (Predict to Plan)

## 1. Benchmark de Modelos de Predição de Trajetória (ML)

| model_name                          |   ade_1s |   fde_1s |   ade_3s |   fde_3s |   ade_5s |   fde_5s |   mean_inference_time_ms |
|:------------------------------------|---------:|---------:|---------:|---------:|---------:|---------:|-------------------------:|
| Constant Velocity (CV)              |    0.712 |    0.712 |    2.932 |    5.151 |    6.024 |   12.208 |                   0      |
| Constant Acceleration (CA)          |    0.496 |    0.496 |    2.14  |    3.785 |    5.629 |   12.606 |                   0.0001 |
| Decision Tree (depth=6)             |    1.459 |    1.459 |    3.65  |    5.842 |    6.495 |   12.185 |                   0.0003 |
| Random Forest (trees=60)            |    0.772 |    0.772 |    2.327 |    3.883 |    4.76  |    9.624 |                   0.0121 |
| Gradient Boosting (HistGB, iter=80) |    0.514 |    0.514 |    1.954 |    3.394 |    4.315 |    9.036 |                   1.1093 |

## 2. Importância das Features (Random Forest)

|                     |          0 |
|:--------------------|-----------:|
| agent_vy            | 0.451784   |
| agent_vx            | 0.431669   |
| agent_speed         | 0.0505959  |
| agent_accel_est     | 0.0202969  |
| past_speed_mean     | 0.00854041 |
| past_dy_1s          | 0.00611023 |
| past_dx_1s          | 0.00513713 |
| yaw_rate_1s         | 0.00463825 |
| past_heading_change | 0.0045978  |
| agent_heading       | 0.00357748 |

## 3. Avaliação em Malha Fechada do Planeador A* com Evasão de Obstáculos

| Planning Modality            |   Scenarios Tested |   Collision Rate (%) |   Min Clearance (m) |   Emergency Decels / Run |   Avg Speed (m/s) |
|:-----------------------------|-------------------:|---------------------:|--------------------:|-------------------------:|------------------:|
| Uninformed (Reactive)        |                 25 |                   16 |                8.9  |                     0    |              3.06 |
| Constant Velocity            |                 25 |                   16 |                8.9  |                     0    |              2.96 |
| ML Predicted (Random Forest) |                 25 |                   16 |                8.93 |                     0.08 |              2.95 |
| Oracle (Ground Truth)        |                 25 |                   16 |                8.9  |                     0    |              3.09 |

![Resultados do Benchmark](../reports/figures/predict_to_plan_benchmark.png)
