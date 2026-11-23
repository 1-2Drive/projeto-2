# Artigo científico — Prever para Planear

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
figuras/             figuras em PDF ou PNG
llncs.cls            classe da Springer (v2.21, 2022-01-12) — não editar
splncs04.bst         estilo de referências da Springer — não editar
build/               PDF e auxiliares (gerado; pode ser apagado)
```

## Secções e páginas previstas

| Ficheiro | Secção | Páginas |
|---|---|---|
| `resumo.tex` | Título, resumo e palavras-chave | 0,5 |
| `introducao.tex` | 1. Introdução | 1 |
| `estado_arte.tex` | 2. Estado da arte | 5,5 |
| `abordagem.tex` | 3. Abordagem proposta | 1,5 |
| `resultados.tex` | 4. Resultados preliminares | 1,5 |
| `conclusao.tex` | 5. Conclusões e trabalho futuro | 0,5 |
| `referencias.bib` | Referências (cerca de 30) | 1,5 |

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
