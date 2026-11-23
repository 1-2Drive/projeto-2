# Prever para Planear: Integração de Modelos de Aprendizagem Automática na Evasão Preditiva de Obstáculos em Veículos Autónomos
**Rascunho do artigo científico (estado da arte e resultados preliminares)**  
*Mestrado em Engenharia de Inteligência Artificial (MEIA) — ISEP*  
*Unidades Curriculares: Aprendizagem Automática & Sistemas de Decisão e Planeamento*

> A versão para entrega está em LaTeX, no formato Springer LNCS, na pasta [`artigo/`](../artigo/) (PDF em `artigo/build/artigo.pdf`). O texto é o mesmo; convém editar lá, para não haver duas versões diferentes. As referências foram todas conferidas e estão em `artigo/referencias.bib`.

---

## Resumo

Um veículo autónomo tem de prever o movimento dos outros utentes da via e de usar essa previsão para planear o seu próprio movimento. Este artigo revê o estado da arte nas duas frentes, a previsão de trajetórias e o planeamento de movimento, e na forma como se ligam e se avaliam, com atenção ao que conseguem os modelos de aprendizagem automática clássica, sem redes neuronais. Apresenta ainda resultados preliminares de uma arquitetura «prever para planear» sobre o Waymo Open Motion Dataset: florestas aleatórias que aprendem, no referencial de cada veículo, a correção a uma extrapolação de velocidade constante, e um planeador de velocidade A\* que trata as previsões como obstáculos móveis. Em 149 cenas reais (13 856 amostras, validação cruzada agrupada por cena), o erro médio de posição desce de 4,97 m para 2,47 m a 3 s e de 12,57 m para 6,66 m a 5 s, face à velocidade constante. No planeamento, a versão atual do planeador ainda não converte esse ganho numa redução mensurável dos conflitos; identificam-se as causas e o trabalho necessário para o conseguir.

**Palavras-chave:** condução autónoma; previsão de trajetórias; aprendizagem automática clássica; planeamento de movimento; A*; Waymo Open Motion Dataset.

---

## 1. Introdução

Os sistemas de condução autónoma organizam-se, em geral, numa cadeia de módulos: perceção, previsão do movimento dos outros utentes da via, planeamento e controlo (Paden et al. 2016; Schwarting, Alonso-Mora, e Rus 2018). A previsão e o planeamento dependem um do outro. Um plano só é seguro se contar com o sítio onde os outros veículos vão estar nos segundos seguintes, e esse futuro não é conhecido: tem de ser estimado a partir do que se observou até ao momento.

Este trabalho junta as duas metades do problema. Na previsão, a questão é quanto se ganha com modelos de aprendizagem automática clássica (árvores de decisão, florestas aleatórias, *gradient boosting*) em relação às extrapolações físicas mais simples. A restrição a modelos clássicos é deliberada: treinam-se em segundos num computador comum, funcionam com poucos dados e deixam ver que variáveis pesam na previsão. No planeamento, a questão é se uma previsão melhor dá um plano melhor, quando as previsões entram como obstáculos móveis num planeador de velocidade baseado em pesquisa A\*.

Os dados são os do Waymo Open Motion Dataset (WOMD) (Ettinger et al. 2021), que regista trajetórias reais de veículos, peões e ciclistas em meio urbano. A escolha do problema foi condicionada pelo que os dados permitem. Cada cena dura cerca de 20 s, o veículo da Waymo percorre nela 96 m (mediana) e, em 82% das cenas analisadas, há um único caminho possível entre o ponto de partida e o de chegada. Não há pedidos de viagem nem frotas. Problemas de rotas longas ou de despacho de veículos exigiriam dados inventados; as interações locais entre veículos, pelo contrário, estão presentes em todas as cenas.

O artigo tem três contributos:

- uma revisão do estado da arte em previsão de trajetórias, em planeamento de movimento e na ligação entre os dois (Secção 2);

- uma arquitetura «prever para planear» com modelos clássicos e um planeador A\* (Secção 3);

- resultados preliminares em 149 cenas reais, com as limitações que ainda têm (Secção 4).

## 2. Estado da arte

### 2.1 Dados e métricas de previsão

A investigação em previsão de movimento apoia-se em conjuntos de dados públicos. O WOMD (Ettinger et al. 2021) tem mais de 100 000 cenas de 20 s, num total de 574 horas a 10 Hz, com as trajetórias de veículos, peões e ciclistas e o mapa da zona. A tarefa oficial dá 1 s de passado e pede os 8 s seguintes. O Argoverse (Chang et al. 2019) tem 324 557 sequências de 5 s, com 2 s observados e 3 s a prever, e distingue-se por publicar, ao lado dos modelos neuronais, referências que não o são: velocidade constante e regressão por vizinhos mais próximos. Em autoestrada, o highD (Krajewski et al. 2018) reúne trajetórias recolhidas por *drone*.

As métricas habituais medem distâncias entre a trajetória prevista e a real (Chang et al. 2019; Ettinger et al. 2021). O erro médio de deslocamento (ADE) é a média dessa distância ao longo do horizonte; o erro final (FDE) é a distância no último instante. Quando o modelo propõe $K$ trajetórias, conta a melhor delas (minADE, minFDE), e a taxa de falhas (*miss rate*) é a fração de casos em que nenhuma fica perto do ponto final real (a menos de 2 m, no Argoverse). Neste trabalho prevê-se uma única trajetória por veículo, pelo que a métrica é o FDE.

### 2.2 Previsão de movimento

A classificação mais citada separa os modelos em três níveis: os físicos, que só usam a dinâmica do veículo; os baseados em manobras, que primeiro reconhecem a intenção do condutor; e os que consideram a interação entre veículos (Lefèvre, Vasquez, e Laugier 2014). Uma revisão mais recente organiza os métodos pela técnica: modelos físicos, aprendizagem automática clássica, aprendizagem profunda e aprendizagem por reforço (Huang et al. 2022). É a segunda família que este trabalho explora.

**Modelos físicos.**

Assumem que o veículo mantém a velocidade, a aceleração ou a taxa de viragem que tem no momento. São os modelos usados no seguimento de veículos com filtros de Kalman, contexto em que foram comparados (Schubert, Richter, e Wanielik 2008). Não precisam de treino, mas só são fiáveis a muito curto prazo, abaixo de um segundo (Lefèvre, Vasquez, e Laugier 2014): ignoram a estrada, os semáforos e os outros veículos.

Ainda assim, são referências difíceis de bater, e por isso qualquer modelo aprendido deve ser comparado com elas. Na previsão de peões, a velocidade constante superou modelos neuronais do estado da arte nos conjuntos de dados habituais (Schöller et al. 2020). Para veículos, a Tabela 1 reúne valores publicados. A distância para os modelos aprendidos cresce com o horizonte, e no WOMD a velocidade constante erra mais de 5 m logo aos 3 s.

| Dados     | Horizonte     | Métrica | Vel. const. | Modelo aprendido      | Fonte                       |
|:----------|:--------------|:--------|:------------|:----------------------|:----------------------------|
| Argoverse | 3 s           | minFDE  | 7,89        | 4,97 (LSTM)           | (Chang et al. 2019)         |
| NGSIM     | 3 s / 5 s     | RMSE    | 3,13 / 6,68 | 2,09 / 4,37 (CS-LSTM) | (Deo e Trivedi 2018)        |
| WOMD      | 3,1 s / 5,1 s | minFDE  | 5,7 / 14,1  | –                     | (Muñoz Sánchez et al. 2022) |
| WOMD      | 8 s           | minADE  | 11,0        | 1,34 (LSTM, K = 6)  | (Ettinger et al. 2021)      |

*Tabela 1. Erros publicados da velocidade constante e de modelos aprendidos na previsão de veículos, em metros, com uma trajetória proposta por veículo, exceto onde se indica K, o número de trajetórias propostas.*

**Aprendizagem automática clássica.**

Antes das redes profundas, a previsão fazia-se com classificadores e regressores sobre variáveis calculadas à mão. Máquinas de vetores de suporte e modelos de Markov escondidos distinguiram condutores que cumprem ou violam a sinalização em cruzamentos, com validação em dados de condução real (Aoude et al. 2012). Em autoestrada, uma floresta aleatória estima a probabilidade de cada manobra e uma regressão por misturas gaussianas converte-a numa distribuição de trajetórias (Schlechtriemen et al. 2015). O estudo mais sistemático nesta linha, com mais de 30 000 km de dados de autoestrada, concluiu que a floresta aleatória classifica as manobras melhor do que um perceptrão multicamada e do que um classificador bayesiano ingénuo, e obteve um erro lateral mediano inferior a 0,21 m a 5 s (Wirthmüller et al. 2021). O Argoverse dá o contraponto: a regressão por vizinhos mais próximos, aplicada diretamente às trajetórias passadas, fica ao nível da velocidade constante (7,88 m contra 7,89 m a 3 s) (Chang et al. 2019). O modelo, por si só, não chega; contam as variáveis que recebe.

As florestas aleatórias (Breiman 2001) e o *gradient boosting* (Friedman 2001) são hoje os modelos de referência para dados em tabela. Estes trabalhos concentram-se, porém, em classificar manobras em contextos específicos, sobretudo a mudança de faixa em autoestrada. Na pesquisa feita para este artigo não se encontrou nenhum que aplique modelos clássicos à regressão de trajetórias no WOMD, em meio urbano.

**Aprendizagem profunda.**

Os melhores resultados atuais vêm de redes neuronais que recebem as trajetórias e o mapa quase em bruto. A VectorNet representa faixas e agentes como vetores ligados num grafo (Gao et al. 2020); o Trajectron++ combina um grafo de agentes com redes recorrentes e garante previsões dinamicamente possíveis (Salzmann et al. 2020); o Motion Transformer, que liderou a tabela do WOMD, atinge um minFDE de 1,22 m com seis trajetórias propostas, em média sobre os horizontes de 3, 5 e 8 s (Shi et al. 2022). Estes números não se comparam diretamente com os de uma previsão única. O preço é o volume de dados, o treino em GPU e a dificuldade em explicar cada previsão.

**Duas ideias que não dependem do modelo.**

A primeira é o referencial. Os modelos centrados no agente, que descrevem a cena a partir da posição e da orientação do veículo a prever, são invariantes a translações e rotações e dominam as tabelas públicas (Su et al. 2022; Shi et al. 2022). A segunda é aprender uma correção em vez da trajetória inteira: parte-se da previsão de um modelo baseado em conhecimento e o modelo aprendido estima apenas o resíduo, o que melhora a exatidão e a generalização a cenas novas (Bahari, Nejjar, e Alahi 2021). Ambas se aplicam a modelos clássicos e são a base da abordagem da Secção 3.

### 2.3 Planeamento de movimento

O planeamento divide-se habitualmente em níveis: a rota na rede viária, a decisão de comportamento (seguir, ceder, mudar de faixa), o planeamento de movimento, que produz uma trajetória, e o controlo (Paden et al. 2016). No planeamento de movimento há três grandes famílias: pesquisa em grafos, amostragem e otimização numérica (Paden et al. 2016).

**Pesquisa em grafos.**

O espaço de estados é discretizado e procura-se o caminho de menor custo. O A\* expande primeiro os estados com menor soma do custo já gasto com uma estimativa do custo em falta; se essa heurística nunca sobrestimar, a solução é ótima (Hart, Nilsson, e Raphael 1968). Para lidar com obstáculos móveis, o tempo e a velocidade passam a fazer parte do estado (Ziegler e Stiller 2009). A decisão longitudinal em meio urbano foi formulada como planeamento discreto resolvido por A\*, com uma heurística construída a partir dos estados de colisão inevitável e um pior caso de 80 ms para um horizonte de 13 s (Hubmann, Aeberhard, e Stiller 2016). O trabalho mais próximo do aqui proposto pesquisa sobre o tempo, a distância percorrida, a faixa e a velocidade discreta, trata os outros veículos como obstáculos no espaço-tempo, previstos com velocidade constante e uma margem de segurança, e usa como heurística o custo de um problema relaxado (Ajanovic et al. 2018).

**Decomposição caminho-velocidade.**

Uma simplificação clássica resolve o problema em dois passos: primeiro um caminho que evita os obstáculos fixos, depois um perfil de velocidade ao longo dele que evita os obstáculos móveis no plano distância-tempo (S–T) (Kant e Zucker 1986). Em estrada, o caminho natural é o eixo da faixa, e as trajetórias descrevem-se em coordenadas de Frenet (distância ao longo da faixa e desvio lateral), onde se geram polinómios que minimizam a variação da aceleração (Werling et al. 2010). O planeador EM do Apollo, de uso industrial, segue esta decomposição: otimiza o caminho e a velocidade em separado, com programação dinâmica para uma solução grosseira e programação quadrática para a refinar (Fan et al. 2018).

**Regras e garantias.**

O Intelligent Driver Model (IDM) calcula a aceleração a partir da distância e da velocidade relativa ao veículo da frente (Treiber, Hennecke, e Helbing 2000); é o modelo que controla os outros veículos nos simuladores atuais (Karnchanachari et al. 2024; Gulino et al. 2023). O modelo RSS define formalmente as distâncias longitudinais e laterais que um veículo deve manter para nunca ser responsável por um acidente (Shalev-Shwartz, Shammah, e Shashua 2017), e pode fundamentar a margem de segurança de um planeador.

### 2.4 Da previsão ao planeamento

**Formas de integração.**

Uma revisão recente distingue três: sequencial, em que a previsão é feita primeiro e entra no planeador como dado de entrada ou termo de custo; não dirigida, em que as duas tarefas são resolvidas em conjunto; e bidirecional, em que o plano influencia a previsão (Hagedorn et al. 2025). A abordagem deste trabalho é sequencial. A sua limitação conhecida é tratar os outros como obstáculos que não reagem, o que torna o veículo defensivo; modelar o efeito das ações próprias nos outros condutores dá comportamentos mais eficientes (Sadigh et al. 2016).

**Uma previsão melhor dá um plano melhor?**

Não necessariamente. A exatidão medida em dados fixos não acompanha de forma fiável o desempenho na condução, por duas razões: o plano do veículo altera o comportamento dos outros, e um preditor mais pesado atrasa o planeador (Tran et al. 2023). As métricas de exatidão tratam todos os erros por igual, quando só alguns mudam o plano, o que motivou métricas que têm em conta o planeamento (Ivanovic e Pavone 2022). O exemplo mais forte é o PDM-Closed, um planeador por regras que prevê os outros veículos com velocidade constante e escolhe entre propostas geradas pelo IDM: no nuPlan obteve 92 e 93 pontos em simulação, contra 72 e 73 do melhor planeador aprendido (com e sem veículos reativos), apesar de ter a pior pontuação quando avaliado apenas contra o registo (Dauner et al. 2023).

**Como se avalia um planeador.**

Em ciclo aberto, compara-se o plano com o que o condutor real fez. Em ciclo fechado, o plano é executado num simulador e o planeador volta a decidir a cada passo. Os outros veículos podem repetir o registo, sem reagir, ou ser controlados por um modelo reativo como o IDM (Karnchanachari et al. 2024). O nuPlan mede colisões por culpa própria, permanência na zona transitável, progresso, tempo até à colisão e conforto (Karnchanachari et al. 2024); o Waymax oferece as mesmas duas opções sobre as cenas do WOMD (Gulino et al. 2023). A repetição do registo tem um defeito conhecido: quando o plano se afasta do que aconteceu, a simulação diverge e surgem colisões com veículos que, na realidade, teriam reagido (Karnchanachari et al. 2024; Hagedorn et al. 2025).

### 2.5 Síntese

Da revisão saem três conclusões que orientam o resto do trabalho. Primeiro, a velocidade constante é a referência obrigatória e o seu erro no WOMD é conhecido (Tabela 1). Segundo, os modelos clássicos têm resultados sólidos na classificação de manobras em autoestrada, mas não foram encontrados trabalhos que os usem para prever trajetórias no WOMD. Terceiro, o ganho em exatidão só interessa se chegar ao plano, e isso mede-se com o planeador, não com o erro de previsão. Tanto quanto foi possível apurar, não existe uma comparação, com um planeador por pesquisa sobre estes dados, entre quatro fontes de previsão: nenhuma, velocidade constante, modelo clássico e futuro real.

## 3. Abordagem proposta

A Figura 1 resume a arquitetura. O código usa a biblioteca scikit-learn (Pedregosa et al. 2011).

```
Cenas do WOMD (20 s, 10 Hz) -> Amostras (veículo, instante) -> Previsão (vel. constante + correção)
    -> Planeador A* (s, v, t) -> Avaliação contra o registo
```

*Figura 1. Arquitetura «prever para planear»: das cenas gravadas ao plano de velocidade do veículo autónomo.*

### 3.1 Dados e amostras

Usam-se dois ficheiros da versão de 20 s do WOMD, com 149 cenas. Uma amostra é um par (veículo, instante). Em cada cena toma-se um instante por segundo e, nele, todos os veículos a menos de 55 m do veículo autónomo, em movimento (pelo menos 0,5 m/s) e observados sem falhas no segundo anterior. O resultado são 13 856 amostras de 2 343 veículos. O alvo é o deslocamento do veículo 1, 3 e 5 s depois; o de 5 s só existe em 10 295 amostras, porque muitos veículos deixam de ser observados antes disso, e as restantes não entram no treino nem na avaliação desse horizonte.

### 3.2 Características e modelos

Cada amostra é descrita por 21 variáveis calculadas apenas com o passado e o presente, em três grupos:

- movimento do próprio veículo, no seu referencial: velocidade longitudinal e lateral, aceleração estimada, taxa de viragem e deslocamento no último segundo;

- veículo da frente: distância, velocidade e velocidade de aproximação, quando existe um veículo ou ciclista na mesma direção, a menos de 60 m e de 1,8 m de desvio lateral (acontece em 66% das amostras);

- relação com o veículo autónomo: posição e velocidade relativas, distância, velocidade de aproximação e tempo até à colisão.

Seguindo as duas ideias da Secção 2.2, o modelo não prevê a posição futura em coordenadas do mapa. Prevê, no referencial do veículo, a diferença entre o deslocamento real e o que a velocidade constante daria:

$$\hat{\mathbf{d}}_h = \mathbf{v}\,h + f_h(\mathbf{x}), \qquad h \in \{1, 3, 5\}\ \mathrm{s}$$

em que $\mathbf{v}$ é a velocidade atual, $\mathbf{x}$ o vetor de características e $f_h$ um modelo treinado para o horizonte $h$. Assim, um veículo que segue em frente tem correção nula seja qual for a direção da estrada no mapa, e o modelo concentra-se nos desvios: travagens, acelerações e curvas. Comparam-se três modelos para $f_h$: árvore de decisão (profundidade 6), floresta aleatória (60 árvores, profundidade 8) e *gradient boosting* por histogramas (80 iterações).

### 3.3 Planeador

O planeador segue a decomposição caminho-velocidade (Secção 2.3): o caminho é dado e planeia-se a velocidade. Um estado é $(s, v, t)$, a distância percorrida, a velocidade e o instante, com passo de 0,1 s e horizonte de 3 s. As ações são quatro acelerações: $-3$, $-1,5$, $0$ e $+1,5$ m/s². O custo de um plano soma o desconforto (quadrado da aceleração) com uma penalização elevada por cada passo a menos de 3,5 m da posição prevista de um veículo; a heurística do A\* premeia o progresso. As posições previstas vêm de uma de quatro fontes: nenhuma (os outros ficam onde estão), velocidade constante, floresta aleatória ou o futuro real, que serve de limite superior.

## 4. Resultados preliminares

Todos os valores desta secção vêm das 149 cenas descritas na Secção 3 e de validação cruzada em cinco partes agrupada por cena: nenhuma cena contribui ao mesmo tempo para o treino e para o teste.

### 4.1 Previsão

| Modelo                    |       1 s       |       3 s       |       5 s        |
|:--------------------------|:---------------:|:---------------:|:----------------:|
| Velocidade constante      | 0,68 ± 0,02 | 4,97 ± 0,35 | 12,57 ± 0,90 |
| Aceleração constante      | 0,44 ± 0,02 | 3,32 ± 0,12 | 9,71 ± 0,35  |
| Floresta aleatória (mapa) | 0,89 ± 0,10 | 4,25 ± 0,37 | 10,02 ± 0,73 |
| Árvore de decisão         | 0,43 ± 0,02 | 2,78 ± 0,14 | 7,50 ± 0,35  |
| Floresta aleatória        | 0,39 ± 0,01 | 2,47 ± 0,12 | 6,66 ± 0,32  |
| *Gradient boosting*       | 0,37 ± 0,01 | 2,36 ± 0,10 | 6,52 ± 0,34  |

*Tabela 2. Erro final de posição (FDE), em metros: média e desvio-padrão das cinco partes da validação cruzada. «(mapa)»: posição prevista diretamente em coordenadas do mapa. Os três últimos modelos usam o referencial do veículo e a correção à velocidade constante.*

A Tabela 2 mostra três coisas. A velocidade constante erra 4,97 m a 3 s e 12,57 m a 5 s, valores coerentes com os 5,7 m e 14,1 m publicados para o mesmo conjunto de dados em horizontes ligeiramente maiores (Tabela 1). A representação conta mais do que o modelo: a mesma floresta aleatória, a prever a posição em coordenadas do mapa, perde para a aceleração constante; no referencial do veículo e a prever a correção, reduz o erro da velocidade constante para cerca de metade a 3 e a 5 s. E os três modelos clássicos ficam próximos, com a floresta aleatória e o *gradient boosting* praticamente empatados.

As variáveis com mais peso na floresta aleatória são a aceleração estimada (0,40), a taxa de viragem (0,15) e a velocidade de aproximação ao veículo da frente (0,09). Sem as três variáveis do veículo da frente, o erro a 3 s sobe de 2,47 m para 2,68 m. Treinar com mais cenas quase não ajuda (2,58 m com 25 cenas, 2,47 m com 119), o que indica que o próximo ganho virá de variáveis novas (mapa, semáforos) e não de mais dados do mesmo tipo.

O mesmo conjunto de variáveis serve para classificar a manobra dos 3 s seguintes em cinco classes (seguir em frente, acelerar, travar, virar à esquerda, virar à direita). Numa divisão simples, com 37 cenas de teste, a floresta aleatória acerta 75,6% e o *gradient boosting* 77,9%; a classe mais frequente representa 55% das amostras.

### 4.2 Planeamento

| Previsão             |    Conflitos (%)    | Travagens máximas | Velocidade (m/s) |
|:---------------------|:-------------------:|:-----------------:|:----------------:|
| Nenhuma              | 21,8 \[16,3; 27,3\] |       0,11        |       5,49       |
| Velocidade constante | 20,7 \[15,6; 25,7\] |       0,02        |       5,38       |
| Floresta aleatória   | 20,2 \[15,1; 25,4\] |       0,02        |       5,36       |
| Futuro real          | 16,7 \[12,0; 21,4\] |       0,04        |       5,37       |

*Tabela 3. Planeador A\* em 909 situações de 121 cenas, por fonte de previsão. Conflito: o plano passa a menos de 3,5 m da posição real de um veículo. Travagens máximas: passos de 0,1 s com a travagem mais forte, em média por situação. Intervalos de confiança a 95% por reamostragem das cenas.*

A Tabela 3 compara as quatro fontes de previsão. Cada situação é um instante de uma cena com pelo menos um veículo à frente do veículo autónomo, a menos de 25 m; em cada parte da validação cruzada, a floresta é treinada nas outras quatro. Conhecer o futuro é melhor do que qualquer das outras fontes: menos 3,5 a 5,1 pontos percentuais de conflitos, com intervalos de confiança que excluem o zero. Planear sem previsão distingue-se sobretudo pelas travagens máximas, cinco vezes mais frequentes. Entre a velocidade constante e a floresta aleatória a diferença é de 0,4 pontos percentuais, com intervalo de confiança de $-0,8$ a $1,9$: não se distinguem. O ganho de exatidão da Tabela 2 não chega, por agora, ao plano, o que está de acordo com a literatura da Secção 2.4.

O valor do futuro real mostra onde está o problema: mesmo com previsões perfeitas há conflitos em 16,7% das situações, pelo que o limite está no planeador e na avaliação, e não nas previsões. Há quatro causas identificadas. O caminho é uma reta na direção atual do veículo, e não a faixa. As posições dos outros veículos, previstas e reais, são interpoladas em linha reta entre o instante atual e os 3 s. O critério de conflito é uma distância entre centros, que conta veículos da faixa do lado; em 3,7% das situações já há um veículo a menos de 3,5 m no instante inicial. E o teste executa um único plano de 3 s em ciclo aberto, sem voltar a planear.

## 5. Conclusões e trabalho futuro

A revisão mostra que as extrapolações físicas continuam a ser referências fortes, que os modelos clássicos têm resultados sólidos em tarefas específicas e que um erro de previsão menor não garante, por si só, um plano melhor. Os resultados preliminares confirmam os dois lados. Com as variáveis certas, uma floresta aleatória reduz para metade o erro da velocidade constante a 3 e a 5 s em cenas urbanas reais. No planeador atual, essa melhoria ainda não se traduz em menos conflitos.

O trabalho seguinte decorre das limitações identificadas:

- planear ao longo da faixa, usando o mapa, com as previsões nos três horizontes em vez de uma interpolação;

- avaliar em ciclo fechado, voltando a planear a cada segundo, com um critério de colisão baseado nas dimensões dos veículos;

- acrescentar variáveis do mapa e dos semáforos à previsão;

- aumentar o número de cenas e separar treino e teste por local, porque o mesmo local aparece em várias cenas;

- comparar o A\* com outras estratégias de pesquisa e medir o número de estados expandidos.

---

## 6. Cronograma e próximos passos
- **Segunda-feira:** apresentação da ideia à docente, fundamentada na análise do dataset e nos resultados preliminares.
- **Quarta-feira:** entrega do rascunho do artigo com o estado da arte.
- **Fase seguinte:** os cinco pontos de trabalho futuro da Secção 5, a começar pelo planeador (planear ao longo da faixa e avaliar em ciclo fechado).

---

**Nota sobre os dados.** Este trabalho foi feito com o Waymo Open Dataset, disponibilizado pela Waymo LLC (*made using the Waymo Open Dataset, provided by Waymo LLC*), ao abrigo da licença para uso não comercial.

---

## Referências

- Ajanovic, Zlatan, Bakir Lacevic, Barys Shyrokau, Michael Stolz, e Martin Horn. 2018. «Search-Based Optimal Motion Planning for Automated Driving». Em *2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, 4523–30. IEEE. <https://doi.org/10.1109/IROS.2018.8593813>.
- Aoude, Georges S., Vishnu R. Desaraju, Lauren H. Stephens, e Jonathan P. How. 2012. «Driver Behavior Classification at Intersections and Validation on Large Naturalistic Data Set». *IEEE Transactions on Intelligent Transportation Systems* 13 (2): 724–36. <https://doi.org/10.1109/TITS.2011.2179537>.
- Bahari, Mohammadhossein, Ismail Nejjar, e Alexandre Alahi. 2021. «Injecting knowledge in data-driven vehicle trajectory predictors». *Transportation Research Part C: Emerging Technologies* 128: 103010. <https://doi.org/10.1016/j.trc.2021.103010>.
- Breiman, Leo. 2001. «Random Forests». *Machine Learning* 45 (1): 5–32. <https://doi.org/10.1023/A:1010933404324>.
- Chang, Ming-Fang, John Lambert, Patsorn Sangkloy, Jagjeet Singh, Slawomir Bak, Andrew Hartnett, De Wang, et al. 2019. «Argoverse: 3D Tracking and Forecasting With Rich Maps». Em *2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 8740–49. IEEE. <https://doi.org/10.1109/CVPR.2019.00895>.
- Dauner, Daniel, Marcel Hallgarten, Andreas Geiger, e Kashyap Chitta. 2023. «Parting with Misconceptions about Learning-based Vehicle Motion Planning». Em *Proceedings of The 7th Conference on Robot Learning*, 229:1268–81. Proceedings of Machine Learning Research. PMLR. <https://proceedings.mlr.press/v229/dauner23a.html>.
- Deo, Nachiket, e Mohan M. Trivedi. 2018. «Convolutional Social Pooling for Vehicle Trajectory Prediction». Em *2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)*, 1549–57. IEEE. <https://doi.org/10.1109/CVPRW.2018.00196>.
- Ettinger, Scott, Shuyang Cheng, Benjamin Caine, Chenxi Liu, Hang Zhao, Sabeek Pradhan, Yuning Chai, et al. 2021. «Large Scale Interactive Motion Forecasting for Autonomous Driving: The Waymo Open Motion Dataset». Em *2021 IEEE/CVF International Conference on Computer Vision (ICCV)*, 9690–99. IEEE. <https://doi.org/10.1109/ICCV48922.2021.00957>.
- Fan, Haoyang, Fan Zhu, Changchun Liu, Liangliang Zhang, Li Zhuang, Dong Li, Weicheng Zhu, Jiangtao Hu, Hongye Li, e Qi Kong. 2018. «Baidu Apollo EM Motion Planner». arXiv:1807.08048.
- Friedman, Jerome H. 2001. «Greedy function approximation: A gradient boosting machine». *The Annals of Statistics* 29 (5): 1189–1232. <https://doi.org/10.1214/aos/1013203451>.
- Gao, Jiyang, Chen Sun, Hang Zhao, Yi Shen, Dragomir Anguelov, Congcong Li, e Cordelia Schmid. 2020. «VectorNet: Encoding HD Maps and Agent Dynamics From Vectorized Representation». Em *2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 11522–30. IEEE. <https://doi.org/10.1109/CVPR42600.2020.01154>.
- Gulino, Cole, Justin Fu, Wenjie Luo, George Tucker, Eli Bronstein, Yiren Lu, Jean Harb, et al. 2023. «Waymax: An Accelerated, Data-Driven Simulator for Large-Scale Autonomous Driving Research». Em *Advances in Neural Information Processing Systems (Datasets and Benchmarks Track)*, 36:7730–42. <https://proceedings.neurips.cc/paper_files/paper/2023/hash/1838feeb71c4b4ea524d0df2f7074245-Abstract-Datasets_and_Benchmarks.html>.
- Hagedorn, Steffen, Marcel Hallgarten, Martin Stoll, e Alexandru Paul Condurache. 2025. «The Integration of Prediction and Planning in Deep Learning Automated Driving Systems: A Review». *IEEE Transactions on Intelligent Vehicles* 10 (5): 3626–43. <https://doi.org/10.1109/TIV.2024.3459071>.
- Hart, Peter E., Nils J. Nilsson, e Bertram Raphael. 1968. «A Formal Basis for the Heuristic Determination of Minimum Cost Paths». *IEEE Transactions on Systems Science and Cybernetics* 4 (2): 100–107. <https://doi.org/10.1109/TSSC.1968.300136>.
- Huang, Yanjun, Jiatong Du, Ziru Yang, Zewei Zhou, Lin Zhang, e Hong Chen. 2022. «A Survey on Trajectory-Prediction Methods for Autonomous Driving». *IEEE Transactions on Intelligent Vehicles* 7 (3): 652–74. <https://doi.org/10.1109/TIV.2022.3167103>.
- Hubmann, Constantin, Michael Aeberhard, e Christoph Stiller. 2016. «A Generic Driving Strategy for Urban Environments». Em *2016 IEEE 19th International Conference on Intelligent Transportation Systems (ITSC)*, 1010–16. IEEE. <https://doi.org/10.1109/ITSC.2016.7795679>.
- Ivanovic, Boris, e Marco Pavone. 2022. «Injecting Planning-Awareness into Prediction and Detection Evaluation». Em *2022 IEEE Intelligent Vehicles Symposium (IV)*, 821–28. IEEE. <https://doi.org/10.1109/IV51971.2022.9827101>.
- Kant, Kamal, e Steven W. Zucker. 1986. «Toward Efficient Trajectory Planning: The Path-Velocity Decomposition». *The International Journal of Robotics Research* 5 (3): 72–89. <https://doi.org/10.1177/027836498600500304>.
- Karnchanachari, Napat, Dimitris Geromichalos, Kok Seang Tan, Nanxiang Li, Christopher Eriksen, Shakiba Yaghoubi, Noushin Mehdipour, et al. 2024. «Towards learning-based planning: The nuPlan benchmark for real-world autonomous driving». Em *2024 IEEE International Conference on Robotics and Automation (ICRA)*, 629–36. IEEE. <https://doi.org/10.1109/ICRA57147.2024.10610077>.
- Krajewski, Robert, Julian Bock, Laurent Kloeker, e Lutz Eckstein. 2018. «The highD Dataset: A Drone Dataset of Naturalistic Vehicle Trajectories on German Highways for Validation of Highly Automated Driving Systems». Em *2018 21st International Conference on Intelligent Transportation Systems (ITSC)*, 2118–25. IEEE. <https://doi.org/10.1109/ITSC.2018.8569552>.
- Lefèvre, Stéphanie, Dizan Vasquez, e Christian Laugier. 2014. «A survey on motion prediction and risk assessment for intelligent vehicles». *ROBOMECH Journal* 1 (1): 1. <https://doi.org/10.1186/s40648-014-0001-z>.
- Muñoz Sánchez, Manuel, Jos Elfring, Emilia Silvas, e René van de Molengraft. 2022. «Scenario-based Evaluation of Prediction Models for Automated Vehicles». Em *25th IEEE International Conference on Intelligent Transportation Systems (ITSC)*, 2227–33. IEEE. <https://doi.org/10.1109/ITSC55140.2022.9922603>.
- Paden, Brian, Michal Čáp, Sze Zheng Yong, Dmitry Yershov, e Emilio Frazzoli. 2016. «A Survey of Motion Planning and Control Techniques for Self-Driving Urban Vehicles». *IEEE Transactions on Intelligent Vehicles* 1 (1): 33–55. <https://doi.org/10.1109/TIV.2016.2578706>.
- Pedregosa, Fabian, Gaël Varoquaux, Alexandre Gramfort, Vincent Michel, Bertrand Thirion, Olivier Grisel, Mathieu Blondel, et al. 2011. «Scikit-learn: Machine Learning in Python». *Journal of Machine Learning Research* 12: 2825–30. <https://www.jmlr.org/papers/v12/pedregosa11a.html>.
- Sadigh, Dorsa, Shankar Sastry, Sanjit A. Seshia, e Anca D. Dragan. 2016. «Planning for Autonomous Cars that Leverage Effects on Human Actions». Em *Proceedings of Robotics: Science and Systems XII*. Ann Arbor, Michigan. <https://doi.org/10.15607/RSS.2016.XII.029>.
- Salzmann, Tim, Boris Ivanovic, Punarjay Chakravarty, e Marco Pavone. 2020. «Trajectron++: Dynamically-Feasible Trajectory Forecasting with Heterogeneous Data». Em *Computer Vision – ECCV 2020*, 12363:683–700. Lecture Notes em Computer Science. Cham: Springer. <https://doi.org/10.1007/978-3-030-58523-5_40>.
- Schlechtriemen, Julian, Florian Wirthmueller, Andreas Wedel, Gabi Breuel, e Klaus-Dieter Kuhnert. 2015. «When will it change the lane? A probabilistic regression approach for rarely occurring events». Em *2015 IEEE Intelligent Vehicles Symposium (IV)*, 1373–79. IEEE. <https://doi.org/10.1109/IVS.2015.7225907>.
- Schöller, Christoph, Vincent Aravantinos, Florian Lay, e Alois Knoll. 2020. «What the Constant Velocity Model Can Teach Us About Pedestrian Motion Prediction». *IEEE Robotics and Automation Letters* 5 (2): 1696–1703. <https://doi.org/10.1109/LRA.2020.2969925>.
- Schubert, Robin, Eric Richter, e Gerd Wanielik. 2008. «Comparison and evaluation of advanced motion models for vehicle tracking». Em *11th International Conference on Information Fusion (FUSION)*, 1–6. IEEE.
- Schwarting, Wilko, Javier Alonso-Mora, e Daniela Rus. 2018. «Planning and Decision-Making for Autonomous Vehicles». *Annual Review of Control, Robotics, and Autonomous Systems* 1: 187–210. <https://doi.org/10.1146/annurev-control-060117-105157>.
- Shalev-Shwartz, Shai, Shaked Shammah, e Amnon Shashua. 2017. «On a Formal Model of Safe and Scalable Self-driving Cars». arXiv:1708.06374.
- Shi, Shaoshuai, Li Jiang, Dengxin Dai, e Bernt Schiele. 2022. «Motion Transformer with Global Intention Localization and Local Movement Refinement». Em *Advances in Neural Information Processing Systems*, 35:6531–43. <https://papers.neurips.cc/paper_files/paper/2022/hash/2ab47c960bfee4f86dfc362f26ad066a-Abstract-Conference.html>.
- Su, DiJia Andy, Bertrand Douillard, Rami Al-Rfou, Cheol Park, e Benjamin Sapp. 2022. «Narrowing the coordinate-frame gap in behavior prediction models: Distillation for efficient and accurate scene-centric motion forecasting». Em *2022 International Conference on Robotics and Automation (ICRA)*, 653–59. IEEE. <https://doi.org/10.1109/ICRA46639.2022.9812368>.
- Tran, Phong, Haoran Wu, Cunjun Yu, Panpan Cai, Sifa Zheng, e David Hsu. 2023. «What Truly Matters in Trajectory Prediction for Autonomous Driving?» Em *Advances in Neural Information Processing Systems*, 36:71327–39. <https://proceedings.neurips.cc/paper_files/paper/2023/hash/e197fe307eb3467035f892dc100d570a-Abstract-Conference.html>.
- Treiber, Martin, Ansgar Hennecke, e Dirk Helbing. 2000. «Congested traffic states in empirical observations and microscopic simulations». *Physical Review E* 62 (2): 1805–24. <https://doi.org/10.1103/PhysRevE.62.1805>.
- Werling, Moritz, Julius Ziegler, Sören Kammel, e Sebastian Thrun. 2010. «Optimal Trajectory Generation for Dynamic Street Scenarios in a Frenét Frame». Em *2010 IEEE International Conference on Robotics and Automation (ICRA)*, 987–93. IEEE. <https://doi.org/10.1109/ROBOT.2010.5509799>.
- Wirthmüller, Florian, Julian Schlechtriemen, Jochen Hipp, e Manfred Reichert. 2021. «Teaching Vehicles to Anticipate: A Systematic Study on Probabilistic Behavior Prediction Using Large Data Sets». *IEEE Transactions on Intelligent Transportation Systems* 22 (11): 7129–44. <https://doi.org/10.1109/TITS.2020.3002070>.
- Ziegler, Julius, e Christoph Stiller. 2009. «Spatiotemporal State Lattices for Fast Trajectory Planning in Dynamic On-Road Driving Scenarios». Em *2009 IEEE/RSJ International Conference on Intelligent Robots and Systems*, 1879–84. IEEE. <https://doi.org/10.1109/IROS.2009.5354448>.
