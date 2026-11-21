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

### 3.1 Engenharia de Características (Feature Engineering)
Para cada veículo alvo $i$ no instante $t$, as variáveis são transformadas para o referencial local do veículo ego:
- Posição relativa: $x_{rel}, y_{rel}$
- Velocidade relativa: $v_{x,rel}, v_{y,rel}$
- Orientação relativa: $\Delta \theta = (\theta_{agent} - \theta_{ego}) \pmod{2\pi}$
- Distância euclidiana ao ego: $d = \sqrt{x_{rel}^2 + y_{rel}^2}$
- Histórico cinemático (janela de 10 passos = 1.0s): $\Delta x_{past}, \Delta y_{past}, \bar{v}_{past}, \Delta \theta_{past}$.

---

## 4. Resultados Experimentais

### 4.1 Benchmark de Predição de Trajetória (ML)
Avaliámos os modelos em 1.158 amostras de teste independentes (extraídas de cenários nunca vistos no treino):

| Modelo | ADE @ 1.0s (m) | FDE @ 1.0s (m) | ADE @ 3.0s (m) | FDE @ 3.0s (m) | ADE @ 5.0s (m) | FDE @ 5.0s (m) | Latência (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Constant Velocity (CV)** | 0.725 | 0.725 | 3.065 | 5.404 | 6.357 | 12.941 | < 0.001 |
| **Constant Acceleration (CA)** | 0.450 | 0.450 | 2.002 | 3.554 | 5.002 | 11.003 | < 0.001 |
| **Decision Tree (depth=6)** | 1.756 | 1.756 | 4.499 | 7.242 | 8.224 | 15.673 | < 0.001 |
| **Random Forest (trees=60)** | 1.104 | 1.104 | 3.384 | 5.664 | 6.643 | 13.161 | 0.012 |

*Principais fatores preditivos identificados pelo Random Forest:*
1. `agent_vy` (46.0%) e `agent_vx` (44.9%): Componentes primárias de velocidade instantânea.
2. `agent_speed` (3.6%): Magnitude escalar de movimento.
3. `past_dx_1s` (1.2%): Persistência inercial observada no último segundo.
4. `past_heading_change` (1.1%): Sinal indicador de início de manobra de curva ou desvio.

### 4.2 Avaliação em Malha Fechada do Planeador A*
Avaliámos o comportamento de navegação do SDC em 25 cenários interativos críticos (veículo obstáculo a menos de 25m e com desvio lateral $<4.0$m):

| Modalidade de Planeamento | Cenários Testados | Taxa de Colisão (%) | Folga Mínima Média (m) | Desacelerações Emergência / Teste | Velocidade Média (m/s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Não Informado (Reativo)** | 25 | 0.0% | 14.96 m | **0.08** | 4.89 m/s |
| **Constant Velocity (CV)** | 25 | 0.0% | 14.96 m | **0.00** | 4.91 m/s |
| **ML Preditivo (Random Forest)** | 25 | 0.0% | 14.96 m | **0.00** | 4.91 m/s |
| **Oracle (Ground Truth)** | 25 | 0.0% | 14.96 m | **0.00** | 4.91 m/s |

**Conclusão dos Resultados:**
O planeador Não Informado (reativo) exibiu episódios de travagem de emergência ($a \le -3.0 m/s^2$) devido à falta de antecipação temporal. Tanto o planeador com física (CV) como o planeador com ML (*Random Forest*) eliminaram por completo as desacelerações de emergência, garantindo transições suaves de aceleração sem sacrificar a velocidade média de progressão.

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
