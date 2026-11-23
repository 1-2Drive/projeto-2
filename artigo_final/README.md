# Artigo científico — Prever para Planear (versão única)

Esta pasta junta os dois rascunhos anteriores: o texto, as referências
conferidas e os resultados de `artigo/`, e as figuras, a tabela de
comparação e a justificação das escolhas de `docs/paper/`, refeitas com o
modelo atual. É a versão a manter; as outras duas ficam como histórico.

Fontes LaTeX do artigo, no formato **Springer LNCS** (o mesmo do template
«Springer Conference Proceedings Template» do Overleaf, versão de
2022-01-12). A organização da pasta segue a do relatório de ENGCIA
(`Relatorio_Previo_ENGCIA/` no repo projeto-1).

**Limite: 12 páginas, referências incluídas.** O `build.sh` avisa quando o
PDF passa esse número.

## Antes de compilar

O título, os autores e a afiliação estão **apenas** em `metadados.tex`.
Confirmar:

| Macro | Verificar |
|---|---|
| `\tituloprincipal`, `\titulocurto` | título completo e título do cabeçalho |
| `\listaautores`, `\autorescurtos`, `\autoresmeta` | elementos do grupo |
| `\emails` | números de aluno, pela mesma ordem dos autores |
| `\palavraschave` | palavras-chave |

O `build.sh` avisa enquanto existirem `TODO` em `metadados.tex`.

## Compilar

Sem LaTeX instalado (caso mais simples): no Overleaf, **New Project →
Upload Project**, carregar um zip desta pasta e escolher `artigo.tex` como
documento principal. A pasta já traz `llncs.cls` e `splncs04.bst`, por isso
não é preciso mais nada.

Com LaTeX instalado (TeX Live, MacTeX ou MiKTeX):

```bash
./build.sh          # PDF em build/artigo.pdf (opções: -w avisos, -c limpar, -h ajuda)
```

Em Windows, o `build.sh` corre no Git Bash ou no WSL. Sem bash:

```
latexmk -pdf -outdir=build artigo.tex
```

## Estrutura

```
artigo.tex           documento principal (a ordem das secções é a dos \input)
metadados.tex        título, autores, afiliação, palavras-chave
preambulo.tex        pacotes, português, ligações, macros
referencias.bib      bibliografia (só aparecem as entradas citadas)
seccoes/             um ficheiro por secção
figuras/             figuras em PDF, geradas por scripts/make_article_figures.py
llncs.cls            classe da Springer (v2.21, 2022-01-12) — não editar
splncs04.bst         estilo de referências da Springer — não editar
build/               PDF e auxiliares (gerado; pode ser apagado)
```

## De onde vêm os números

| O quê | Script |
|---|---|
| Tabela de previsão, figuras, classificação de manobras | `python scripts/make_article_figures.py --out artigo_final/figuras` |
| Regressão linear, vizinhos mais próximos, SVM, curva de aprendizagem | `python scripts/compare_alternative_models.py` |
| Tabela do planeador | `python scripts/evaluate_planner_cv.py` |

Os números do artigo foram obtidos com 149 cenas de dois ficheiros
`training_20s` (extraídas com as variáveis do veículo da frente). O
«segundo conjunto» é o `data/processed/predict_to_plan_samples.csv` do
repositório (178 cenas, sem essas variáveis). Correr os scripts sobre
outra tabela muda os números: atualizar o texto em conformidade.

## Secções e páginas

| Ficheiro | Secção | Páginas |
|---|---|---|
| `resumo.tex`, `introducao.tex` | Resumo e 1. Introdução | 1,5 |
| `estado_arte.tex` | 2. Estado da arte | 4,5 |
| `abordagem.tex` | 3. Abordagem proposta | 1,5 |
| `resultados.tex` | 4. Resultados preliminares | 2 |
| `conclusao.tex` | 5. Conclusões e trabalho futuro | 0,5 |
| `referencias.bib` | Referências (31 citadas) | 2 |

O PDF tem 12 páginas e a última está quase cheia: para acrescentar texto
é preciso cortar noutro sítio.

## Macros próprias

| Comando | Uso |
|---|---|
| `\preencher{texto}` | informação em falta (realçada a laranja); não pode sobrar nenhuma na entrega |
| `\un{2,5}{m}` | número com unidade, sem quebra de linha |
| `\var{lead_gap}` | nome de uma variável do código |
| `\cref{...}` | referências cruzadas («Figura 1», «Secção 2.2», «Tabela 3») |
| `$1,5$` | vírgula decimal em modo matemático (pacote `icomma`) |

## Regras do formato

- Não mudar margens, tamanho de letra nem espaçamentos: é o que define o
  formato e o limite de páginas.
- Citações com `\cite{chave}`; a chave é a do `referencias.bib`. O estilo
  `splncs04` é numérico e ordena as referências por apelido.
- Qualquer figura ou resultado feito com os dados da Waymo precisa da
  atribuição pedida pela licença (está em `seccoes/agradecimentos.tex`).
- Os ficheiros de dados da Waymo não entram no repositório.
