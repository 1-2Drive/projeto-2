# Inspeção do Dataset: Waymo Open Motion (Predict-to-Plan)

Este painel apresenta a estrutura, contratos de dados, estatísticas e amostras reais do dataset processado utilizado no projeto.

## 1. Visão Geral dos Ficheiros Gerados

| Ficheiro | Amostras | Cenários Únicos | Dimensão | Finalidade |
| :--- | :---: | :---: | :---: | :--- |
| `data/processed/train.csv` | **4,267** | 124 | 3.1 MB | Treino dos regressores e classificadores |
| `data/processed/val.csv` | **1,075** | 27 | 791 KB | Afinação de hiperparâmetros e validação |
| `data/processed/test.csv` | **800** | 27 | 591 KB | Avaliação cega e métricas finais |
| `data/processed/predict_to_plan_samples.csv` | **6,142** | 178 | 4.5 MB | Dataset total consolidado |

## 2. Dicionário de Colunas e Contratos Formais

O dataset contém **44 colunas** divididas em 4 grupos funcionais rigorosos:

### A. Metadados e Rastreabilidade (Excluídos de X)
- `scenario_id`: Identificador único do episódio (20 segundos a 10 Hz)
- `track_id`: Identificador do veículo/obstáculo dentro do cenário
- `object_type`: Tipo de agente (1 = Veículo, 2 = Peão, 3 = Ciclista)
- `current_time_step`: Passo temporal de referência no cenário

### B. Coordenadas Globais (Excluídas de X para garantir invariância espacial)
- `ego_x`, `ego_y`, `ego_heading`, `ego_speed`: Estado cinemático absoluto do veículo ego
- `agent_x`, `agent_y`, `agent_vx`, `agent_vy`, `agent_speed`, `agent_heading`: Estado do obstáculo

### C. Vetor de Características Relativas (X in R^21)
- `rel_x`, `rel_y`: Posição relativa do obstáculo no referencial do ego
- `rel_vx`, `rel_vy`: Velocidades relativas
- `distance_to_ego`: Distância euclidiana instantânea
- `closing_speed`: Taxa de aproximação ao longo da linha de visão
- `ttc`: Time-To-Collision vetorial (em segundos; 99.0 se a divergir)
- `azimuth_angle`: Ângulo azimutal no referencial frontal do ego
- `longitudinal_gap`, `lateral_gap`: Separação nos eixos de Frenet
- `past_dx_1s`, `past_dy_1s`, `past_speed_mean`, `past_heading_change`, `yaw_rate_1s`, `agent_accel_est`: Histórico inercial recente

### D. Targets de Predição (Y)
- `future_dx_1s`, `future_dy_1s`: Deslocamento a 1.0 segundo
- `future_dx_3s`, `future_dy_3s`: Deslocamento a 3.0 segundos (horizonte principal)
- `future_dx_5s`, `future_dy_5s`: Deslocamento a 5.0 segundos
- `maneuver_class`: Classe de manobra tática (STRAIGHT, TURN_LEFT, TURN_RIGHT, DECELERATING, ACCELERATING)

## 3. Distribuição das Classes de Manobra (Treino)
| Classe de Manobra | Contagem de Amostras | Percentagem |
| :--- | :---: | :---: |
| `STRAIGHT` | 2,283 | 53.5% |
| `DECELERATING` | 650 | 15.2% |
| `ACCELERATING` | 608 | 14.2% |
| `TURN_LEFT` | 400 | 9.4% |
| `TURN_RIGHT` | 326 | 7.6% |

## 4. Amostra Real dos Dados (Primeiras 5 Linhas de `train.csv`)

| scenario_id | track_id | agent_speed | distance_to_ego | closing_speed | ttc | future_dx_3s | future_dy_3s | maneuver_class |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 28fe360951cf98d6 | 0 | 1.45 | 13.08 | 0.48 | 26.99 | 6.79 | 6.51 | TURN_RIGHT |
| 28fe360951cf98d6 | 1 | 4.44 | 40.49 | -0.80 | 99.00 | -18.38 | 1.64 | TURN_LEFT |
| 28fe360951cf98d6 | 0 | 2.84 | 13.64 | -1.17 | 99.00 | 10.26 | 5.74 | TURN_RIGHT |
| 28fe360951cf98d6 | 1 | 5.23 | 41.52 | -1.62 | 99.00 | -22.09 | 1.27 | ACCELERATING |
| 28fe360951cf98d6 | 0 | 3.55 | 15.26 | -2.03 | 99.00 | 13.84 | 3.50 | TURN_RIGHT |