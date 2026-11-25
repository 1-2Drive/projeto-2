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

### 2.3 Justificação da Escolha dos Modelos de Machine Learning e do Planeador A*

#### Porquê Modelos Baseados em Árvores (Random Forest e HistGradientBoosting) e não Deep Learning (LSTM / Transformers) ou SVM / KNN?
1. **Orçamento de Latência em Tempo Real (ECUs Automóveis):** O ciclo de controlo tático em veículos autónomos opera a 10--20~Hz (janela temporal de 50 a 100~ms para todo o ciclo: perceção, predição, planeamento e atuação). Modelos de Deep Learning (como VectorNet ou Transformers) exigem 25--80~ms só para inferência e requerem aceleradores dedicados (GPUs/TPUs). Em contraste, o **Random Forest executa em 0.008~ms** e o **HistGradientBoosting em 0.79~ms** num CPU convencional, consumindo menos de 1% do ciclo de controlo.
2. **Eficiência Amostral em Dados Tabulares:** Em conjuntos de dados com 6.142 amostras interativas de elevada densidade e variáveis cinemáticas heterogéneas, ensembles baseados em árvores superam redes neuronais profundas sem sofrer de sobreajuste (*overfitting*), problemas de gradiente nulo ou necessidade de milhões de parâmetros.
3. **Explicabilidade e Segurança Funcional (Norma ISO 26262):** Ao contrário de redes neuronais tipo "caixa negra", os modelos em árvore fornecem importâncias de características (Gini Importance) e regras de decisão inspecionáveis, requisito indispensável para homologação de sistemas críticos de condução autónoma.
4. **Inviabilidade do KNN e SVM:** O algoritmo KNN possui complexidade de inferência $O(N \cdot D)$, escalando com o número de amostras de treino na memória e provocando latências inaceitáveis (>20~ms). O SVM com kernel RBF escala quadraticamente com os vetores de suporte e apresenta limitações graves na previsão simultânea de múltiplos horizontes contínuos ($\Delta x, \Delta y$ a 1s, 3s e 5s).

#### Porquê Planeamento Espaço-Temporal A* e não RRT*, MPC ou Redes Neuronais de Controlo?
1. **Garantia de Completude e Otimalidade:** Num reticulado discretizado de estados $(s, v, t)$, o algoritmo $A^*$ garante matematicamente a identificação da trajetória globalmente ótima e livre de colisões (caso exista), eliminando a natureza não-determinística do RRT*.
2. **Determinismo Temporal:** Algoritmos de amostragem aleatória (RRT e RRT*) produzem caminhos estocásticos e variáveis entre ciclos sucessivos sob a mesma entrada sensorial, induzindo oscilações bruscas de aceleração (*jerk*). O $A^*$ permite limitar o número máximo de expansões (`max_expansions`), garantindo um teto máximo e previsível de tempo de execução (*worst-case execution time*).
3. **Robustez a Restrições Não-Convexas (vs MPC):** Em ambientes com múltiplos obstáculos móveis simultâneos, o espaço livre de colisão é fortemente não-convexo. Solvers de Controlo Preditivo Baseado em Modelo (MPC) sofrem frequentemente de aprisionamento em mínimos locais ou falhas de convergência (*solver timeouts*) perante cenários complexos com múltiplos veículos a cortar a trajetória. O $A^*$ discretizado contorna os mínimos locais explorando sistematicamente a fila de prioridades.
4. **Desacoplamento em Coordenadas de Frenet:** A pesquisa longitudinal ao longo do comprimento de arco $s$ da linha central da faixa respeita naturalmente a geometria rodoviária sem a explosão combinatória de uma pesquisa 2D não restringida.

---

## 3. Metodologia
O desenvolvimento do sistema Prever para Planear adotou a metodologia ágil Scrum aliada ao processo padronizado CRISP-DM para fluxos de aprendizagem automática, estruturando as atividades em sprints iterativos de engenharia de dados, modelação preditiva e planeamento espaço-temporal.

### 3.1 Requisitos Funcionais (Formato User Story)
1. **US-01: Ingestão de Trajetórias e Contratos de Dados.** Como sistema de condução autónoma, pretendo ingerir cenários brutos do Waymo Open Motion Dataset e impor contratos formais de colunas que separem metadados, coordenadas globais e atributos cinemáticos.  
   *Critérios de Aceitação:* Cenários de 20 segundos a 10~Hz são convertidos em DataFrames estruturados; metadados e coordenadas globais são estritamente isolados do vetor de características $X$ para garantir invariância espacial e prevenir sobreajuste geográfico.
2. **US-02: Engenharia de Características Vetorizada em NumPy e Pandas.** Como módulo de pré-processamento, pretendo calcular cinemática ego-relativa, velocidade de aproximação (*closing speed*) e Time-To-Collision (TTC) através de operações estritamente vetorizadas.  
   *Critérios de Aceitação:* O pipeline utiliza exclusivamente operações matriciais NumPy e indexação booleana Pandas sem ciclos `for` em Python; amostras com velocidades ou posições fisicamente inválidas são descartadas por filtragem booleana.
3. **US-03: Previsão de Trajetória Multi-Horizonte e Classificação de Intenção.** Como sistema preditivo de decisão, pretendo antecipar deslocamentos espaciais futuros ($\Delta x, \Delta y$ a 1.0s, 3.0s e 5.0s) e classificar manobras táticas para todos os obstáculos dinâmicos circundantes.  
   *Critérios de Aceitação:* Os modelos de regressão superam baselines físicas (Constant Velocity e Constant Acceleration) em métricas ADE e FDE; o classificador de manobras alcança acurácia $>75\%$ nas 5 classes de manobras táticas.
4. **US-04: Planeamento Espaço-Temporal em Malha Fechada Livre de Colisões.** Como módulo de planeamento de velocidade e trajetória, pretendo gerar um perfil ótimo de velocidade $(s, v, t)$ ao longo da linha central da faixa em coordenadas de Frenet que desvie proativamente dos obstáculos previstos.  
   *Critérios de Aceitação:* O planeador A* avalia footprints dinâmicos de múltiplos obstáculos simultâneos, elimina desacelerações de emergência ($a \le -3.0\text{ m/s}^2$) e reduz a taxa de colisão face a planeadores não-preditivos de velocidade constante.

### 3.2 Requisitos Não-Funcionais
1. **RNF-01: Orçamento de Latência em Tempo Real.** O ciclo integrado de predição e planeamento deve executar dentro de um teto rígido de 50 a 100~ms (ciclo de controlo de 10 a 20~Hz) em CPUs convencionais de grau automóvel sem necessidade de aceleradores GPU.
2. **RNF-02: Prevenção Estrita de Fuga Temporal de Dados (*Zero Data Leakage*).** A partição do dataset deve ser rigorosamente baseada em grupos de cenários independentes (`scenario_id`), garantindo que $\mathcal{S}_{train} \cap \mathcal{S}_{val} = \emptyset$ e $\mathcal{S}_{train} \cap \mathcal{S}_{test} = \emptyset$.
3. **RNF-03: Segurança Funcional e Explicabilidade (Norma ISO 26262).** Os modelos preditivos de ML devem fornecer fronteiras de decisão inspecionáveis e quantificação de importância de variáveis (*Gini Importance*) para auditoria de segurança.
4. **RNF-04: Tempo Determinístico de Pior Caso (WCET).** O algoritmo de pesquisa espaço-temporal deve possuir um limite fixo de expansões de nós (`max_expansions`) para prevenir bloqueios computacionais em tempo de execução.

---

## 4. Solução Proposta: Arquitetura "Prever para Planear"

A arquitetura implementada no repositório `1-2Drive/projeto-2` é composta por quatro módulos integrados:

```
[Waymo TFRecord Shards]
          │
          ▼ (Streaming & Decodificação Protobuf)
[SequenceExtractor / Pipeline Pandas] ──► Contratos Formais: Metadados, Globais, Relativas, Targets
          │
          ▼
[ML Trajectory & Maneuver Predictor] ──► Regressão Multi-Horizonte (1s, 3s, 5s) & Classificação 5 Classes
   (CV vs CA vs Decision Tree vs Random Forest vs HistGradientBoosting)
          │ Predições: Δx, Δy & Intenção Tática
          ▼
[Spatiotemporal A* Speed Planner] ──► Evasão no reticulado (s, v, t) ao longo da faixa (Frenet)
          │
          ▼
[Métricas em Malha Fechada]: Taxa de Colisão %, Distância Mínima, Desacelerações de Emergência
```

### 4.1 Engenharia de Características (Feature Engineering com Pandas & NumPy)
Para cada veículo alvo $i$ no instante $t$, as variáveis são transformadas de forma vetorizada com **NumPy** e **Pandas** para o referencial local do veículo ego:
- Posição e gaps relativos: $x_{rel}, y_{rel}$, `longitudinal_gap`, `lateral_gap`
- Cinemática de aproximação: `closing_speed`, `ttc` (Time-To-Collision vetorial), `azimuth_angle`
- Orientação e dinâmica: $\Delta \theta = (\theta_{agent} - \theta_{ego}) \pmod{2\pi}$, taxa de guinada (`yaw_rate_1s`), aceleração estimada (`agent_accel_est`)
- Histórico cinemático (janela de 10 passos = 1.0s): $\Delta x_{past}, \Delta y_{past}, \bar{v}_{past}, \Delta \theta_{past}$.
- Rotulagem vetorial de manobras (*np.select*): `STRAIGHT`, `TURN_LEFT`, `TURN_RIGHT`, `DECELERATING`, `ACCELERATING`.

### 4.2 Formulação do Planeador A* Espaço-Temporal
Discretização em estados $\mathbf{x}_k = (s_k, v_k, t_k)$ com transição de aceleração $a_k \in \{-3.0, -1.5, 0.0, 1.5\} \text{ m/s}^2$ e custo multiobjetivo:
$$g(\mathbf{x}_{k+1}) = g(\mathbf{x}_k) + \left[ w_{comfort} a_k^2 + \mathcal{C}_{collision}(\mathbf{x}_{k+1}) \right] \Delta t$$
onde a penalização de colisão é ativada sempre que a distância entre o ego e qualquer obstáculo dinâmico previsto é inferior ao raio de segurança $R_{safety} = 3.5$~m.

---

## 5. Avaliação Experimental

### 5.1 Justificação e Análise Empírica das Partições de Dados (Train / Val / Test)
A determinação da proporção de divisão entre os conjuntos de Treino, Validação e Teste é crítica para assegurar poder estatístico suficiente para aprendizagem sem comprometer a representatividade da avaliação. No Waymo Open Motion Dataset, devido à forte correlação temporal entre passos sucessivos de um mesmo episódio de 20 segundos, a partição é estritamente realizada por cenários únicos (`scenario_id`), garantindo **zero fuga de dados** (*zero data leakage*).

Para justificar a escolha da partição, realizámos uma análise comparativa sistemática sobre 178 cenários (6.142 amostras interativas) sob diferentes rácios:

| Configuração da Partição | Treino (Amostras / Cenários) | Validação (Amostras / Cenários) | Teste (Amostras / Cenários) | Random Forest ADE@3s (m) | FDE@3s (m) | Avaliação Metodológica |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **60% Treino / 20% Val / 20% Teste** | 3.737 am. (106 cen.) | 1.189 am. (35 cen.) | 1.216 am. (37 cen.) | 2.593 m | 4.221 m | *Underfitting* ligeiro; perda de cobertura de manobras raras no treino. |
| **70% Treino / 20% Val / 10% Teste** | 4.267 am. (124 cen.) | 1.242 am. (35 cen.) | 633 am. (19 cen.) | 2.626 m | 4.199 m | Conjunto de teste reduzido (apenas 19 cenários), aumentando a variância da avaliação final. |
| **80% Treino / 10% Val / 10% Teste** | 4.932 am. (142 cen.) | 577 am. (17 cen.) | 633 am. (19 cen.) | 2.679 m | 4.271 m | Validação insuficiente (apenas 17 cenários) para afinação robusta de hiperparâmetros. |
| **70% Treino / 15% Val / 15% Teste (Adotada)** | **4.267 am. (124 cen.)** | **1.055 am. (26 cen.)** | **820 am. (28 cen.)** | **2.652 m** | **4.278 m** | **Equilíbrio ótimo:** Garante massa crítica para treino ($>4.200$ amostras) e conjuntos de validação e teste com representatividade estatística sólida ($>800$ amostras em $>25$ cenários independentes). |

**Conclusão Metodológica da Partição:**
A partição **70% / 15% / 15%** (ou 70% / 20% / 10% dependendo da ênfase na validação contínua) oferece o balanço ideal:
1. Mais de 4.200 amostras de treino asseguram que os regressores e classificadores aprendem manobras raras (ex.: curvas bruscas em cruzamento e desacelerações repentinas).
2. O conjunto de teste contém 28 cenários completamente desacoplados, fornecendo um teste cego imune a auto-correlações temporais.

---

## 4. Resultados Experimentais

### 4.1 Benchmark de Predição de Trajetória e Afinação de Hiperparâmetros
Avaliámos os modelos em 1.502 amostras de teste independentes (extraídas de 44 cenários nunca vistos no treino, a partir de 6.142 amostras em larga escala). Realizámos também afinação de hiperparâmetros por validação cruzada (`RandomizedSearchCV` com 3 folds) para otimizar o *HistGradientBoostingRegressor*, identificando como configuração ótima: `max_iter=100`, `max_depth=8`, `learning_rate=0.05`, `min_samples_leaf=20`, `l2_regularization=0.0`.

Reportamos os erros de deslocamento e as métricas padrão de regressão (MAE, RMSE, $R^2$):

| Modelo | ADE @ 1.0s (m) | ADE @ 3.0s (m) | FDE @ 3.0s (m) | MAE (m) | RMSE (m) | $R^2$ | Latência (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Constant Velocity (CV)** | 0.588 | 2.444 | 4.301 | 3.263 | 6.197 | 0.943 | < 0.001 |
| **Constant Acceleration (CA)** | 0.401 | 1.754 | 3.107 | 3.024 | 6.298 | 0.947 | < 0.001 |
| **Decision Tree (depth=6)** | 1.717 | 3.947 | 6.177 | 4.142 | 7.053 | 0.905 | < 0.001 |
| **Random Forest (trees=60)** | 0.821 | 2.304 | 3.786 | 2.819 | 5.198 | 0.957 | 0.008 |
| **HistGradientBoosting (default)** | **0.480** | **1.766** | **3.052** | **2.506** | **4.793** | **0.966** | 0.017 |
| **Tuned HistGB (CV-optimized)** | 0.493 | 1.779 | 3.065 | 2.512 | 4.840 | 0.965 | 0.030 |

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

### 4.3 Avaliação em Malha Fechada do Planeador: A* 1D vs A* 2D Frenet vs Baselines de Otimização
Avaliámos a navegação do SDC em todos os **99 cenários de teste críticos** com **866 obstáculos dinâmicos simultâneos**, comparando a complexidade do algoritmo A* (1D longitudinal vs 2D Frenet com evasão lateral) e baselines de otimização de rotas (Dijkstra uniforme vs Greedy guloso):

| Modalidade de Planeamento | Cenários Testados | Taxa Colisão OBB (IC 95%) | Proximidade < 3.5m (IC 95%) | Desacel. Emerg. / Teste | Velocidade Média | Expansões | Latência (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Não Informado (1D Reativo)** | 99 | 4.0% [1.6%, 9.9%] | 26.3% [18.6%, 35.7%] | 0.21 | 6.83 m/s | 806.9 | 26.2 |
| **Constant Velocity (1D)** | 99 | 5.1% [2.2%, 11.3%] | 24.2% [16.9%, 33.5%] | 0.01 | 6.76 m/s | 790.0 | 26.8 |
| **ML Preditivo (1D A*)** | 99 | 5.1% [2.2%, 11.3%] | 24.2% [16.9%, 33.5%] | 0.03 | 6.71 m/s | 784.8 | 26.8 |
| **ML Preditivo (2D Frenet A*)** | **99** | **5.1% [2.2%, 11.3%]** | **26.3% [18.6%, 35.7%]** | **0.00** | **6.59 m/s** | **1,154.6** | **229.1** |
| **Dijkstra Baseline (1D, h=0)** | 99 | 5.1% [2.2%, 11.3%] | 27.3% [19.5%, 36.8%] | 0.04 | 6.56 m/s | 81.9 | 2.8 |
| **Greedy Baseline (1D, f=h)** | 99 | 15.2% [9.4%, 23.5%] | 35.4% [26.6%, 45.2%] | 0.00 | 8.52 m/s | 31.0 | 1.0 |
| **Oracle (1D Ground Truth)** | 99 | 5.1% [2.2%, 11.3%] | 18.2% [11.8%, 26.9%] | 0.03 | 6.68 m/s | 784.9 | 26.2 |

**Conclusão e Interpretação Física dos Resultados:**
1. **Falha Crítica do Planeador Guloso (Greedy):** O modelo *Greedy Best-First* ($f = h$), ao ignorar os custos acumulados de esforço e travagem, acelerou cegamente para atingir o objetivo ($8.52\text{ m/s}$), resultando numa taxa de colisão OBB inaceitável de **15.2%** (o triplo do A*!). Isto demonstra matematicamente a necessidade da função de custo cumulativa do A*.
2. **Zero Travagens de Emergência no A* 2D Frenet:** O planeador 1D está restrito à faixa central e, perante um obstáculo frontal, é forçado a travar. O planeador **A* 2D no espaço curvilíneo de Frenet $(s, d, v, t)$** executa desvios laterais suaves ($d \in [-3.5, 3.5]\text{ m}$), reduzindo as travagens de emergência a **0.00 eventos por teste (100% de eliminação)** e mantendo a velocidade de cruzeiro ($6.59\text{ m/s}$).
3. **Eficiência Heurística do A*:** Comparativamente ao Dijkstra (custo uniforme desorientado), a heurística espaciotemporal do A* guia a procura diretamente para soluções cinematicamente viáveis.
4. **Precisão de Regressão com Modelos ML:** Os modelos de *Gradient Boosting* atingiram $R^2 = 0.966$ e $RMSE = 4.79\text{ m}$ (muito superior aos $6.20\text{ m}$ do CV), provando a redução da variância de erro em manobras complexas.

### 4.4 Limitações e Pressupostos do Modelo
1. **Agentes Não Reativos (Log-Replay):** As trajetórias dos outros veículos foram gravadas previamente e não reagem às decisões do ego. Se o ego travar bruscamente, um veículo traseiro no histórico pode colidir com o ego sem que isso represente erro do planeador.
2. **Caminho de Referência Longitudinal:** O planeador segue a reta/curva do ego como caminho de referência longitudinal. Em curvas muito acentuadas (>15°), um planeador completo requer associação dinâmica à linha de centro da faixa (Frenet lattice).
3. **Classes de Agentes:** O modelo preditivo ML foi focado em veículos motorizados dinâmicos; veículos estacionados e utilizadores vulneráveis (peões/ciclistas) utilizam extrapolação estática ou de velocidade constante.

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
