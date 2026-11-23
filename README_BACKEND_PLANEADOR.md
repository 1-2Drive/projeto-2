# Backend: fidelidade do planeador A* e da avaliação em malha fechada

> **Estado:** diagnóstico e proposta, ainda por implementar.
> **Ficheiros em causa:** [scripts/run_predict_to_plan.py](scripts/run_predict_to_plan.py), [src/planning/astar_planner.py](src/planning/astar_planner.py) e [src/preprocessing/sequence_extractor.py](src/preprocessing/sequence_extractor.py).
> **Relacionado:** [README_UI_3D.md](README_UI_3D.md). A UI 3D vai tornar estes problemas visíveis, mas não depende da sua correção.

## Resumo

- **O benchmark de planeamento (tabela 4 do artigo) não mede bem o que o artigo diz que mede.**
  - O ego é planeado numa **reta**, e não na estrada.
  - O "futuro real" dos obstáculos também é uma **reta**.
  - A colisão é um **círculo de 3,5 m** que apanha carros na faixa ao lado.
  - Os carros parados, os peões e os ciclistas são **invisíveis**.
- **As diferenças entre modos não são estatisticamente distinguíveis.** Os modos diferem em 1 a 3 cenas num total de 30, e os intervalos de confiança sobrepõem-se quase por completo.
- **Proposta:** usar o caminho real do ego, as posições reais dos obstáculos a 10 Hz e a sobreposição de retângulos com as dimensões reais. Incluir todos os agentes relevantes, avaliar todas as cenas e reportar intervalos de confiança. Depois, voltar a correr o benchmark e atualizar o artigo.

---

## 1. Como funciona hoje

O pipeline está em [run_predict_to_plan.py](scripts/run_predict_to_plan.py):

1. **Extração:** o `SequenceExtractor` gera uma amostra por obstáculo e por instante `t` = 1, 2, …, 6 s (*stride* de 10 passos a 10 Hz).
2. **Divisão:** as amostras são divididas em treino e teste por cenário (25 % para teste).
3. **Seleção:** as amostras de teste são agrupadas por `(scenario_id, t)` e fica-se só com as cenas que têm um obstáculo "crítico" à frente ([linha 166](scripts/run_predict_to_plan.py#L166)). Destas, avaliam-se só as **primeiras 30** ([linha 175](scripts/run_predict_to_plan.py#L175)).
4. **Planeamento:** em cada cena, o A\* escolhe um perfil de **velocidade** (aceleração em {−3; −1,5; 0; +1,5} m/s²) ao longo de um caminho fixo, com um horizonte de 3 s. Fá-lo para cada um dos 4 modos de previsão.
5. **Avaliação:** o plano é comparado com o "futuro real" dos obstáculos. Há colisão se a distância entre centros for inferior a 3,5 m.

---

## 2. Problemas

### P1. O caminho do ego é uma reta, e não a estrada

```python
# run_predict_to_plan.py:184-188
path_length = 60.0
ego_path = ego_xy + s * (cos(ego_heading), sin(ego_heading))   # reta de 60 m
```

O A\* **não escolhe a direção**: só decide a velocidade ao longo deste caminho. Quando o ego real vira (num cruzamento, numa rotunda ou numa curva), o plano é avaliado num caminho que o carro nunca percorreu.

- Ignora os obstáculos que o ego real encontraria na curva.
- Conta colisões com veículos que estão na reta mas fora do caminho real.

**Dimensão do problema**, nos 178 cenários processados:

| Variação de heading do ego em 5 s | % de cenários |
| :--- | ---: |
| > 15° | 13,5 % |
| > 45° | 5,6 % |

**Contradição com o artigo:** o [main.tex](docs/paper/main.tex), na secção sobre a escolha do A\*, diz que o planeador respeita *"road topology along the Frenet centerline"*. O código não usa nenhuma linha central de faixa.

```text
              │ ▲ ego real (vira à esquerda)
              │ │
   ───────────┘ │
   ego ●═══════╪════════════════►  caminho onde o A* planeia e é avaliado
                    🚗 obstáculo em frente → conta como "colisão"
```

### P2. O "futuro real" dos obstáculos é uma reta

```python
# run_predict_to_plan.py:204-209
gx = ox0 + future_dx_3s * (step / 30)   # reta, a velocidade constante, até à posição a +3 s
```

- **A verdade usada é uma aproximação.** Uma travagem, uma curva ou uma aceleração a meio do horizonte desaparecem. Esta mesma reta serve para avaliar as colisões de **todos** os modos.
- **A previsão RF desperdiça informação.** Usa só o ponto a 3 s e interpola-o em linha reta ([linhas 221-223](scripts/run_predict_to_plan.py#L221-L223)). Ignora as previsões a 1 s e a 5 s, que o modelo já calcula.

### P3. A colisão é um círculo de 3,5 m entre centros

```python
# astar_planner.py:134 e :206
if dist_to_obs < self.safety_radius:   # 3,5 m entre centros, sem dimensões dos veículos
```

- **Os veículos são tratados como pontos.** Ignoram-se o comprimento (~4,5–5 m), a largura (~2 m) e a orientação.
- **3,5 m é quase a largura de uma faixa (3,5–3,7 m).** Um carro na faixa ao lado, a circular lado a lado, conta como colisão.
- **Nos dados:**
  - **167** amostras têm um obstáculo lado a lado na faixa adjacente (`|rel_y|` entre 2,5 e 4,5 m e `|rel_x|` < 5 m);
  - **26** obstáculos já estão a menos de 3,5 m **no instante inicial**, ou seja, colisões que nenhum planeador pode evitar.

Isto explica a maior parte dos **13,3 % de colisões do Oráculo**. Como o Oráculo prevê exatamente a mesma reta com que é avaliado, as suas colisões vêm sobretudo deste critério e de o A\* só poder travar ou acelerar.

### P4. Há obstáculos invisíveis ao planeador e à avaliação

O `SequenceExtractor` ([sequence_extractor.py:124-142](src/preprocessing/sequence_extractor.py#L124-L142)) só guarda:
- **veículos** (`object_type == 1`);
- com velocidade **≥ 1,0 m/s** (`min_agent_speed=1.0` em [run_predict_to_plan.py:67](scripts/run_predict_to_plan.py#L67));
- a uma distância **≤ 50 m** do ego.

Por isso, o planeador e a avaliação ignoram:
- os **carros parados**, incluindo filas de trânsito à frente do ego;
- os **peões** e os **ciclistas**;
- os **semáforos** e os sinais **STOP**.

Um plano que atravesse um carro parado ou um peão não conta como colisão.

### P5. A amostra é pequena e não há testes estatísticos

São avaliadas só 30 cenas, por isso uma cena vale **3,3 pontos percentuais**. Os intervalos de confiança a 95 % (Wilson) são:

| Modo | Colisões | Taxa | IC 95 % |
| :--- | ---: | ---: | :--- |
| Oráculo | 4/30 | 13,3 % | 5,3 % – 29,7 % |
| ML (RF) | 5/30 | 16,7 % | 7,3 % – 33,6 % |
| Reativo | 6/30 | 20,0 % | 9,5 % – 37,3 % |
| Vel. Constante | 7/30 | 23,3 % | 11,8 % – 40,9 % |

A frase *"the predictive planner reduces collisions from 23.3 % to 16.7 %"* corresponde a **2 cenas**. Os intervalos sobrepõem-se quase por completo, por isso não se pode afirmar que um modo é melhor do que outro. Além disso, avaliam-se só as **primeiras** 30 cenas válidas, e não todas.

### P6. O artigo e o código não batem certo

| Afirmação | Onde | Realidade |
| :--- | :--- | :--- |
| *"along the Frenet centerline"* | [main.tex](docs/paper/main.tex), escolha do A\* | Caminho reto ao longo do heading (P1) |
| *"reduces collisions from 23.3 % (CV) to 16.7 %"* | Abstract do [main.tex](docs/paper/main.tex) | Diferença de 2 cenas, sem significância estatística (P5) |
| *"completely eliminating emergency decelerations (0.30 → 0.00)"* | Abstract do [main.tex](docs/paper/main.tex) | O 0,30 é do modo **Reativo**, não do CV, que já tinha 0,00. A frase junta dois baselines diferentes |
| *"guarantees zero emergency braking"* (US-04) | [main.tex](docs/paper/main.tex), user stories | Não travar a fundo não é, por si, uma garantia de segurança: o RF tem 0 travagens e 16,7 % de colisões |
| *"navegação suave e sem colisões"* | Resumo de [artigo_prever_para_planear.md](docs/artigo_prever_para_planear.md) | 16,7 % de colisões |
| *"cenários de 20 segundos"* | [README.md](README.md) e o artigo em markdown | Os CSVs só têm `t` até 6 s com 3 s de futuro, o que só é compatível com cenários de ~9,1 s (91 passos) |

---

## 3. Impacto

- **Tabela 4 e conclusões do artigo:** as taxas de colisão medem sobretudo artefactos (P1 a P4) e ruído amostral (P5), e não a qualidade da previsão.
- **UI 3D:** o ego "fantasma" atravessaria passeios nas curvas, e muitas das "colisões" mostradas seriam carros na faixa ao lado.
- **Defesa:** quem olhar para o código ou para a visualização encontra estas inconsistências facilmente. É melhor corrigi-las ou declará-las como limitações antes de alguém as apontar.

---

## 4. Proposta de correção

Proponho fazer a correção em fases, cada uma verificável.

### F1. Extrair a lógica por cena, sem mudar o comportamento

Mover a lógica por cena de [run_predict_to_plan.py](scripts/run_predict_to_plan.py) (linhas ~164–242) para um módulo novo, `src/planning/scene_planning.py`, com estas funções:
- `select_critical_scenes`;
- `build_ego_path`;
- `build_obstacle_predictions`;
- `plan_scene`.

**Critério:** a tabela 4 é reproduzida **sem diferenças**. Este passo também é necessário para a UI 3D.

### F2. Extrair cenas completas a partir dos TFRecords

Hoje, o pipeline só guarda uma linha por obstáculo, com deslocamentos agregados. Para avaliar a sério, é preciso guardar a cena inteira. Proponho um `SceneExtractor` novo que, por `(scenario_id, t)`, guarde:
- a trajetória **gravada** do ego entre `t` e `t + horizonte`, a 10 Hz;
- para cada agente relevante: tipo, dimensões, a posição atual, o histórico (que alimenta as features do RF), o futuro real a 10 Hz e a máscara de validade.

A cache fica em `data/processed/planning_scenes.pkl`, porque o CSV não serve para estruturas aninhadas.

### F3. Caminho de referência real (resolve P1)

O caminho do A\* passa a ser a **trajetória gravada do ego** (avaliação *log-replay*). É a prática habitual em benchmarks de planeamento sobre dados reais. Se a trajetória gravada for mais curta do que o horizonte, é prolongada em reta.

**Alternativa:** a linha central da faixa a partir do mapa, que seria um verdadeiro Frenet. É mais fiel ao texto atual do artigo, mas exige associar o ego à sequência de faixas certa, o que dá bastante mais trabalho.

### F4. Futuro real a 10 Hz e previsão RF por troços (resolve P2)

- **Oráculo e avaliação:** usam as posições reais a 10 Hz. Os passos inválidos ficam de fora.
- **RF:** a previsão passa a interpolar por troços entre 0, 1, 3 e 5 s, em vez de seguir uma reta até aos 3 s.

### F5. Colisão por retângulos orientados (resolve P3)

- **Ego e obstáculos:** passam a ser retângulos com o comprimento, a largura e o heading reais, e a sobreposição é testada pelo teorema do eixo separador (SAT).
- **Planeador:** acrescenta uma margem de segurança configurável (por exemplo, 0,5 m) à função de custo.
- **Avaliação:** considera só a sobreposição real.
- **Sobreposições no instante inicial:** são reportadas à parte, como "inevitáveis", e não contam como falha do planeador.

### F6. Incluir todos os agentes relevantes (resolve P4)

- Veículos parados, peões e ciclistas passam a ser obstáculos. Os parados usam uma previsão estática.
- O RF só foi treinado com veículos, por isso os peões e os ciclistas usam CV. Isto deve ser dito no artigo.
- **Ressalva do log-replay:** os agentes gravados não reagem ao ego. Se o ego travar, o carro de trás "bate". As colisões passam a ser classificadas em **frontal, lateral e traseira**, e as traseiras com o ego parado ou a travar ficam como "não imputáveis ao ego".

### F7. Estatística (resolve P5)

- Avaliar **todas** as cenas válidas de teste, e não só as primeiras 30. Se possível, usar também mais shards.
- Reportar a taxa de colisão com **IC 95 % de Wilson**.
- Comparar os modos com o **teste de McNemar** (emparelhado, porque todos os modos são avaliados nas mesmas cenas).

### F8. Atualizar o artigo e a documentação (resolve P6)

- Tabela 4, abstract e conclusões do [main.tex](docs/paper/main.tex) e do [artigo_prever_para_planear.md](docs/artigo_prever_para_planear.md).
- Descrever o caminho de referência como *"logged ego trajectory (log-replay)"*, ou implementar mesmo o Frenet.
- Corrigir "20 segundos" para ~9,1 s no [README.md](README.md).
- Acrescentar uma subsecção "Limitations" com os agentes não reativos e a previsão CV para peões e ciclistas.

---

## 5. Verificação

1. **Regressão (F1):** `python scripts/run_predict_to_plan.py` reproduz a tabela 4 atual sem diferenças.
2. **Testes unitários novos:**
   - sobreposição de retângulos (casos lado a lado sem colisão, em T e rodados);
   - interpolação por troços do RF;
   - extração do caminho gravado do ego, incluindo passos inválidos e o prolongamento em reta;
   - IC de Wilson e teste de McNemar com valores conhecidos.
3. **Teste de sanidade:** com o Oráculo e as colisões classificadas, a taxa de colisões **imputáveis ao ego** deve ser ~0 %. Se não for, há um bug no planeador ou na avaliação.
4. **Inspeção visual:** abrir na UI 3D ([README_UI_3D.md](README_UI_3D.md)) algumas cenas de cada tipo de colisão e confirmar que fazem sentido.

---

## 6. Decisões em aberto (para a equipa)

| Decisão | Opções | Recomendação |
| :--- | :--- | :--- |
| Caminho de referência | Trajetória gravada do ego / linha central da faixa (Frenet) | Trajetória gravada, ajustando o texto do artigo |
| Colisões traseiras não imputáveis | Excluir / reportar à parte / contar | Reportar à parte |
| Horizonte de planeamento | 3 s (atual) / 5 s | Manter 3 s e usar a previsão a 5 s só como contexto |
| Volume de avaliação | Shards atuais / mais shards | Mais shards, se o tempo permitir; os IC dizem se chega |
| Corrigir antes da entrega ou declarar como limitação | Corrigir (F1–F8) / declarar as limitações e manter os números | Corrigir pelo menos F1, F3, F5 e F7 |

---

## 7. Outro problema do backend: credencial no repositório

O [scripts/download_all_shards.py](scripts/download_all_shards.py) tem um **token de acesso SharePoint escrito no código**, que já está no histórico do git. Esse token expirou a 21/11/2026, mas continua exposto e identifica a conta institucional. Proposta:
- ler o token de uma variável de ambiente (`SHAREPOINT_TOKEN`) e documentá-la no `.env.example`;
- se o repositório for ou vier a ser público, limpar o histórico com `git filter-repo`.
