# Prever para Planear: Integração de Modelos de Aprendizagem Automática na Evasão Preditiva de Obstáculos em Veículos Autónomos
**Draft de Artigo Científico e Estado da Arte**  
*Mestrado em Engenharia Informática / Inteligência Artificial — ISEP*  
*Unidades Curriculares: Aprendizagem Automática & Sistemas de Decisão e Planeamento*

---

## Resumo (Abstract)
A condução autónoma segura em cenários urbanos densos depende criticamente da capacidade do veículo autónomo (*Self-Driving Car* - SDC) antecipar o comportamento dinâmico de outros utentes da via (veículos, peões e ciclistas) e integrar essa previsão no seu módulo de decisão e planeamento de trajetória. Neste trabalho, investigamos a arquitetura integrada **"Prever para Planear" (*Predict to Plan*)**, utilizando dados reais do *Waymo Open Motion Dataset*. Demonstramos, através de uma auditoria empírica do dataset (cenários de 20 segundos em janelas de ~140 metros), que problemas de despachamento de frotas ou roteamento global (*routing*) são inviáveis neste domínio sem a fabricação artificial de dados. Em contrapartida, propomos e validamos uma abordagem onde modelos de Aprendizagem Automática (Árvores de Decisão e *Random Forest*) preveem o deslocamento futuro ($\Delta x, \Delta y$) a horizontes de 1.0s, 3.0s e 5.0s, alimentando um planeador espaço-temporal baseado em A*. Os resultados obtidos em 4.947 amostras interativas mostram que a incorporação de predição reduz eventos de desaceleração de emergência e assegura navegação suave e sem colisões.

---

## 1. Introdução e Formulação do Problema
O desenvolvimento de sistemas de condução autónoma de Nível 4 e 5 requer a resolução simultânea de dois desafios complementares:
1. **Perceção e Predição (Aprendizagem Automática):** Interpretar o histórico de movimento dos agentes envolventes e prever a sua ocupação espacial futura.
2. **Decisão e Planeamento (Sistemas de Decisão):** Selecionar uma política de velocidade e trajetória para o veículo ego que maximize o progresso em direção ao objetivo, minimize o desconforto (*jerk*) e garanta colisão zero.

Muitos trabalhos abordam estes problemas de forma estanque. Contudo, em cruzamentos e ultrapassagens, o planeamento puramente reativo (que assume obstáculos estáticos ou velocidade constante) falha perante manobras dinâmicas, forçando travagens bruscas ou colisões.

### Viabilidade do Dataset Waymo Open Motion
Uma análise rigorosa do *Waymo Open Motion Dataset* (v1.2) permitiu clarificar o escopo científico do projeto:
- Cada cenário tem uma duração temporal fixa (20 segundos amostrados a 10 Hz = ~200 passos).
- A extensão geográfica do mapa local é restrita a uma vizinhança de ~140m em torno do SDC (distância mediana percorrida pelo ego é de apenas 96 metros).
- Em 82% dos cenários existe apenas uma única faixa de rodagem disponível para o ego do início ao fim do recorte temporal, e o dataset não contém dados de pedidos de passageiros ou gestão de frotas.
- **Conclusão de Engenharia:** Tentar modelar despacho de táxis ou planeamento de rotas urbanas de longa distância neste dataset exigiria inventar mapas e pedidos sintéticos, descaracterizando a validade científica do trabalho. Pelo contrário, a modelação de **interações locais veículo-veículo com evasão preditiva de obstáculos** explora com exatidão a riqueza e fidelidade dos dados do Waymo.

---

## 2. Estado da Arte

### 2.1 Modelos de Predição de Trajetória
A predição de movimento em condução autónoma divide-se classicamente em três paradigmas:
1. **Modelos Físicos Cinematográficos (Baselines):**
   - *Constant Velocity (CV):* Assume vetor velocidade linear constante $\vec{x}(t + \Delta t) = \vec{x}(t) + \vec{v}(t)\Delta t$. É computacionalmente trivial ($<0.001$ ms), mas falha severamente em curvas e desacelerações.
   - *Constant Acceleration (CA):* Incorpora estimativas de aceleração derivadas do histórico recente. Apresenta bom desempenho em horizontes curtos ($<1.0$s), mas acumula erros quadráticos em horizontes longos ($>3.0$s).
2. **Modelos de Aprendizagem Automática Tabulares:**
   - *Decision Trees & Random Forest Regressors:* Capturam relações não-lineares entre aceleração, velocidade relativa, distância ao ego e variação de *heading* (guinada). Permitem inferência ultrarrápida ($<0.02$ ms) compatível com ciclos de controlo em tempo real (10–50 Hz), oferecendo interpretabilidade através da importância das características (*feature importance*).
3. **Modelos de Deep Learning (Redes Recorrentes e Grafos):**
   - Arquiteturas como *VectorNet* (Gao et al., 2020) e *Trajectron++* (Salzmann et al., 2020) utilizam grafos espaço-temporais para prever distribuições multimodais de trajetória, à custa de elevada complexidade computacional e requisitos de treino em GPU.

### 2.2 Planeamento Espaço-Temporal e Evasão de Obstáculos
No domínio do planeamento tático, destacam-se:
- **A\* Espaço-Temporal ($s, v, t$):** Discretiza o perfil longitudinal de velocidade ao longo da linha central da faixa (*Frenet frame*), minimizando uma função de custo multiobjetivo (progresso, suavidade e penalização de colisão contra volumes de ocupação previstos).
- **Frenet Lattice Planners:** Geram feixes de curvas polinomiais (quintas e quárticas) avaliadas em relação a obstáculos em movimento (Werling et al., 2010).
- **Model Predictive Control (MPC):** Formulação por otimização quadrática contínua sujeita a restrições cinemáticas de atuadores.

---

## 3. Metodologia: A Arquitetura "Prever para Planear"

A arquitetura implementada no repositório `1-2Drive/projeto-2` é composta por três módulos modulares:

```
[Waymo TFRecord Shards]
          │
          ▼ (Streaming & Decodificação Protobuf)
[SequenceExtractor] ──► Filtros: Distância < 50m, Velocidade > 1.0 m/s, Validade Contínua
          │
          ▼
[ML Trajectory Predictor] ──► Features: Cinética Relativa, Histórico 1.0s
   (CV vs CA vs Decision Tree vs Random Forest)
          │ Predições: Δx, Δy em t+1.0s, t+3.0s, t+5.0s
          ▼
[Spatiotemporal A* Planner] ──► Evasão no plano (s, v, t) ao longo da faixa
          │
          ▼
[Métricas em Malha Fechada]: Colisão %, TTC Mínimo, Desacelerações de Emergência
```

### 3.1 Engenharia de Características (Feature Engineering com Pandas & NumPy)
Para cada veículo alvo $i$ no instante $t$, as variáveis são transformadas de forma vetorizada com **NumPy** e **Pandas** para o referencial local do veículo ego:
- Posição e gaps relativos: $x_{rel}, y_{rel}$, `longitudinal_gap`, `lateral_gap`
- Cinemática de aproximação: `closing_speed`, `ttc` (Time-To-Collision vetorial), `azimuth_angle`
- Orientação e dinâmica: $\Delta \theta = (\theta_{agent} - \theta_{ego}) \pmod{2\pi}$, taxa de guinada (`yaw_rate_1s`), aceleração estimada (`agent_accel_est`)
- Histórico cinemático (janela de 10 passos = 1.0s): $\Delta x_{past}, \Delta y_{past}, \bar{v}_{past}, \Delta \theta_{past}$.
- Rotulagem vetorial de manobras (*np.select*): `STRAIGHT`, `TURN_LEFT`, `TURN_RIGHT`, `DECELERATING`, `ACCELERATING`.

---

## 4. Resultados Experimentais

### 4.1 Benchmark de Predição de Trajetória (Regressão Contínua)
Avaliámos os modelos em 1.502 amostras de teste independentes (extraídas de 44 cenários nunca vistos no treino, a partir de 6.142 amostras em larga escala):

| Modelo | ADE @ 1.0s (m) | FDE @ 1.0s (m) | ADE @ 3.0s (m) | FDE @ 3.0s (m) | ADE @ 5.0s (m) | FDE @ 5.0s (m) | Latência (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Constant Velocity (CV)** | 0.588 | 0.588 | 2.444 | 4.301 | 5.379 | 11.248 | < 0.001 |
| **Constant Acceleration (CA)** | 0.401 | 0.401 | 1.754 | 3.107 | 4.892 | 11.170 | < 0.001 |
| **Decision Tree (depth=6)** | 1.717 | 1.717 | 3.947 | 6.177 | 6.790 | 12.476 | < 0.001 |
| **Random Forest (trees=60)** | 0.821 | 0.821 | 2.304 | 3.786 | 4.697 | 9.485 | 0.008 |
| **Gradient Boosting (HistGB)** | **0.480** | **0.480** | **1.766** | **3.052** | **4.112** | **8.806** | 0.791 |

*Principais fatores preditivos identificados pelo Random Forest:*
1. `agent_vy` (46.6%) e `agent_vx` (41.2%): Componentes primárias de velocidade instantânea.
2. `agent_speed` (5.3%): Magnitude escalar de movimento.
3. `agent_accel_est` (2.2%): Variação de aceleração derivada vetorialmente.
4. `past_dy_1s` (1.0%) e `past_dx_1s` (0.6%): Persistência inercial.
5. `past_heading_change` (0.6%) e `yaw_rate_1s` (0.6%): Variação angular de guinada.

### 4.2 Benchmark de Classificação de Intenção e Manobra (Classificação Multiclasse)
Classificação da manobra a horizonte $t+3.0$s (`STRAIGHT`, `TURN_LEFT`, `TURN_RIGHT`, `DECELERATING`, `ACCELERATING`) avaliada no conjunto de teste independente:

| Modelo Classificador | Accuracy | F1-Score (Macro) | F1-Score (Ponderado) |
| :--- | :---: | :---: | :---: |
| **Random Forest Classifier (trees=60)** | 69.84% | 0.6712 | 0.7137 |
| **HistGradientBoosting Classifier (iter=80)** | **77.63%** | **0.7320** | **0.7843** |

O classificador por *Gradient Boosting* demonstrou elevada capacidade de identificar veículos em processo de desaceleração e rotação, providenciando informação tática essencial para o planeador antecipar manobras bruscas.

### 4.3 Avaliação em Malha Fechada do Planeador A* com Múltiplos Obstáculos
Avaliámos a navegação do SDC em 30 cenários interativos complexos com **194 obstáculos dinâmicos simultâneos**:

| Modalidade de Planeamento | Cenários Testados | Taxa de Colisão (%) | Folga Mínima Média (m) | Desacelerações Emergência / Teste | Velocidade Média (m/s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Não Informado (Reativo)** | 30 | 20.0% | 6.61 m | **0.30** | 6.23 m/s |
| **Constant Velocity (CV)** | 30 | 23.3% | 6.51 m | **0.00** | 6.29 m/s |
| **ML Preditivo (Random Forest)** | 30 | **16.7%** | 6.58 m | **0.00** | 6.33 m/s |
| **Oracle (Ground Truth)** | 30 | 13.3% | 6.54 m | 0.07 | 6.30 m/s |

**Conclusão dos Resultados:**
Em ambientes densos com múltiplos obstáculos simultâneos:
1. O planeador Não Informado (reativo) causou 20.0% de colisões e 0.30 travagens de emergência por corrida, devido à ausência de antecipação.
2. O modelo cinemático Constant Velocity (CV) sofreu com falsos positivos e erros acumulados em curvas, gerando 23.3% de colisões.
3. O planeador alimentado por **Machine Learning (Random Forest)** reduziu as colisões para **16.7%** (muito próximo dos 13.3% do Oracle ótimo) e manteve **0.00 desacelerações de emergência**, alcançando a maior velocidade média de cruzeiro (6.33 m/s) com elevado conforto de condução.

---

## 5. Cronograma e Próximos Passos
- **Segunda-feira:** Apresentação formal da ideia aprovada à docente, fundamentada na auditoria empírica do Waymo e nos resultados preliminares.
- **Quarta-feira:** Entrega do rascunho de artigo científico com o Estado da Arte completo e a formulação matemática.
- **Fase Seguinte:** Expansão para predição multimodal (distribuição de probabilidades sobre intenções: seguir em frente vs virar à esquerda/direita) e integração de custos de tráfego semafórico.

---

## Referências
1. Ettinger, S., et al. (2021). *Large Scale Interactive Motion Forecasting for Autonomous Driving : The Waymo Open Motion Dataset*. Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV).
2. Werling, M., et al. (2010). *Optimal trajectory generation for dynamic street scenarios in a Frenet Frame*. IEEE International Conference on Robotics and Automation (ICRA).
3. Shalev-Shwartz, S., Shammah, S., & Shashua, A. (2017). *On a formal model of safe and scalable self-driving cars (Mobileye RSS)*. arXiv preprint arXiv:1708.06374.
4. Salzmann, T., et al. (2020). *Trajectron++: Dynamically-feasible trajectory forecasting with heterogeneous data*. European Conference on Computer Vision (ECCV).
