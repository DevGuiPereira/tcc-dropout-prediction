# Relatório Consolidado

**TCC — Desenvolvimento de modelo preditivo interpretável de machine learning para identificação de
alunos em risco de evasão no ensino superior**
Curso de Sistemas de Informação — UEMG

> **Convenção de partição — vale para todas as tabelas deste documento.** Salvo indicação em
> contrário, todo número de desempenho vem da **partição da semente 42**. A
> **Verificação 2** (seção 10) mostrou que essa partição é **favorável**: em partições típicas
> o F1 é ≈ 0,88 (não 0,9059), o modelo detecta **84 de cada 100** evadidos (não 90, faixa 82–86) e
> perde 44 em vez de 29. A precisão não apresenta esse viés. **Comparações entre modelos permanecem
> válidas**, porque todos são avaliados sobre exatamente os mesmos alunos.

---

> **O que este documento é.** O registro único e completo de tudo o que foi construído e de todas as
> conclusões alcançadas. Ele é autocontido: não é necessário abrir nenhum outro arquivo para entender o
> que foi feito, o que se descobriu e o que o trabalho pode ou não afirmar.
>
> A documentação detalhada por fase está em [`docs/`](docs/) (13 documentos). Este relatório é a
> consolidação — e a única fonte que reúne também as **três verificações finais** (calibração,
> significância, equidade) e a **revisão de coerência interna** que se seguiu a elas, que revisaram
> parcialmente conclusões anteriores.
>
> **Esta é a versão final antes da escrita da monografia.** O changelog da última rodada de correções
> está em [`docs/12-revisao-final.md`](docs/12-revisao-final.md).

## Índice

1. [O problema e o objetivo](#1-o-problema-e-o-objetivo)
2. [O que foi construído](#2-o-que-foi-construído)
3. [Fase 2 — Entendimento dos dados](#3-fase-2--entendimento-dos-dados)
4. [A falha de integridade encontrada no banco](#4-a-falha-de-integridade-encontrada-no-banco)
5. [Fase 3 — Preparação dos dados](#5-fase-3--preparação-dos-dados)
6. [Fase 4 — Modelagem](#6-fase-4--modelagem)
7. [Fase 5 — Interpretabilidade](#7-fase-5--interpretabilidade)
8. [Fase 6 — Consolidação e ferramenta de uso](#8-fase-6--consolidação-e-ferramenta-de-uso)
9. [Verificação 1 — Calibração de probabilidade](#9-verificação-1--calibração-de-probabilidade)
10. [Verificação 2 — Significância estatística](#10-verificação-2--significância-estatística)
11. [Verificação 3 — Equidade algorítmica](#11-verificação-3--equidade-algorítmica)
12. [Conclusões consolidadas](#12-conclusões-consolidadas)
13. [O que o trabalho não sustenta](#13-o-que-o-trabalho-não-sustenta)
14. [Trabalhos futuros](#14-trabalhos-futuros)
15. [Erros cometidos e corrigidos](#15-erros-cometidos-e-corrigidos)
16. [Como reproduzir](#16-como-reproduzir)

---

## 1. O problema e o objetivo

A evasão no ensino superior custa ao aluno a formação interrompida, à instituição a receita e a vaga
ociosa, e ao sistema educacional o investimento desperdiçado. Na base analisada, **32,1% dos 4.424
estudantes evadiram**.

A dificuldade prática não é reconhecer a evasão depois que ocorre, e sim **identificar com antecedência
quem está em risco** e — sobretudo — **por quê**, já que a intervenção adequada depende da causa.

### A pergunta de pesquisa em duas partes

| Pergunta | Onde é respondida |
|---|---|
| **Quem está em risco de evadir?** | Fases 3 e 4 — questão preditiva |
| **Por quê?** | Fase 5 — questão explicativa, com valores SHAP |

A segunda pergunta é a que confere utilidade prática. Dois alunos com o mesmo escore de risco podem
demandar intervenções de setores diferentes: um sinalizado por inadimplência precisa da assistência
estudantil; outro por reprovações precisa de apoio pedagógico. Sem explicação individual, a coordenação
recebe os dois na mesma lista e trata os dois do mesmo jeito.

### Dados

| | |
|---|---|
| **Dataset** | Predict Students' Dropout and Academic Success — UCI id **697** (Realinho et al., 2022) |
| **Contexto** | Instituto Politécnico de Portalegre, Portugal — ingressos de 2008 a 2019 |
| **Registros** | 4.424 estudantes |
| **Atributos** | 36 variáveis independentes + variável-alvo |
| **Fonte primária** | Banco MySQL local (`tcc.data`), com fallback automático para o UCI |
| **Alvo** | `Graduate` 49,9% · `Dropout` 32,1% · `Enrolled` 17,9% |

---

## 2. O que foi construído

### Notebooks

| Notebook | Fase / função | Células |
|---|---|---:|
| `01_analise_exploratoria.ipynb` | Fase 2 — entendimento dos dados | 48 |
| `02_preprocessamento.ipynb` | Fase 3 — preparação | 27 |
| `03_treinamento_modelos.ipynb` | Fase 4 — modelagem | 44 |
| `04_interpretabilidade_shap.ipynb` | Fase 5 — interpretabilidade | 49 |
| `05_relatorio_final.ipynb` | Fase 6 — consolidação + ferramenta | 23 |
| `06_calibracao.ipynb` | Verificação — calibração | 15 |
| `07_significancia.ipynb` | Verificação — significância estatística | 18 |
| `08_auditoria_equidade.ipynb` | Verificação — equidade algorítmica | 14 |

### Artefatos

| Diretório | Conteúdo |
|---|---|
| `src/data_loader.py` | Carga MySQL/UCI, padronização de colunas, correção de integridade, rótulos em português |
| `dados/` | 24 arquivos — CSVs intermediários, metadados, métricas, valores SHAP, resultados das verificações |
| `modelos/` | 6 arquivos — pré-processador, os três modelos, o melhor modelo, metadados |
| `figuras/` | **29 PNGs** em 200 dpi |
| `docs/` | **13 documentos** markdown, ~48 mil palavras |

### Ambiente

Python 3.14.4 · pandas 3.0.5 · scikit-learn 1.9.0 · xgboost 3.4.0 · imbalanced-learn 0.14.2 ·
shap 0.52.0 · statsmodels 0.14.6 · MySQL Server 8.4.8

`random_state=42` em toda operação estocástica, exceto na análise de variabilidade por semente, onde a
variação é o próprio objeto.

---

## 3. Fase 2 — Entendimento dos dados

### 3.1 A aparência enganosa de "tudo numérico"

O `dtypes` mostra 29 colunas inteiras e 7 de ponto flutuante — aparentemente todas numéricas. **Falso:**
9 das 36 colunas são **códigos de categoria**. `course = 9254` não é maior nem melhor que
`course = 171`; são cursos diferentes. Tratá-los como grandezas contínuas ensinaria ao modelo uma
ordenação inexistente.

A tipologia foi declarada manualmente (19 contínuas, 8 binárias, 9 nominais), com base na documentação
dos autores. **Nenhuma inferência automática distinguiria `course` de `age`** — ambos são inteiros.

### 3.2 Qualidade

| Verificação | Resultado |
|---|---|
| Valores ausentes | **0** em 4.424 × 37 |
| Duplicatas | **0** |
| Colunas constantes | **nenhuma** |

Dispensa imputação, o que elimina uma fonte de viés. Em contrapartida, o dataset é mais limpo do que a
realidade de um sistema acadêmico brasileiro — o pipeline **não foi exposto ao problema de dados
faltantes**.

### 3.3 Onde está o sinal

**Correlações mais fortes com a evasão** (ponto-bisserial):

| Variável | r |
|---|---:|
| Média das notas (2º sem.) | −0,572 |
| Disciplinas aprovadas (2º sem.) | −0,570 |
| Média das notas (1º sem.) | −0,481 |
| Disciplinas aprovadas (1º sem.) | −0,479 |
| Mensalidades em dia | −0,429 |
| Idade no ingresso | +0,254 |

**Taxa de evasão por variável binária** (taxa-base 32,1%):

| Variável | Valor 0 | Valor 1 | Diferença |
|---|---:|---:|---:|
| **Mensalidades em dia** | **86,6%** | 24,7% | **−61,8 pp** |
| Inadimplente | 28,3% | 62,0% | +33,7 pp |
| **Bolsista** | 38,7% | **12,2%** | **−26,5 pp** |
| Gênero | 25,1% | 45,1% | +19,9 pp |

> **Os dois números mais expressivos da análise exploratória:** entre alunos com mensalidades em atraso
> a evasão é de **86,6%**, contra 24,7% dos adimplentes — 62 pontos percentuais em uma única variável
> binária. E **bolsistas evadem a 12,2%**, menos de um terço dos não bolsistas.

### 3.4 Separabilidade e a decisão sobre `Enrolled`

| Variável | `Dropout` | `Enrolled` | `Graduate` |
|---|---:|---:|---:|
| Disciplinas aprovadas (2º sem.) | **0,0** | 4,0 | 6,0 |
| Média das notas (2º sem.) | **0,0** | 12,0 | 13,0 |
| Idade no ingresso | 23,0 | 20,0 | 19,0 |

Dois achados:

1. **A mediana de aprovações no 2º semestre é zero entre os evadidos e seis entre os concluintes.** Não
   é diferença de grau — é de natureza. O aluno para de aprovar disciplinas antes de formalizar a saída.
2. **`Enrolled` fica sistematicamente no meio**, sem região própria. É o padrão esperado de um rótulo
   que mistura futuros formandos com futuros evadidos.

### 3.5 Outliers: são o sinal, não ruído

**2.422 registros (54,7%)** têm ao menos um outlier pelo método IQR.

| | Taxa de evasão |
|---|---:|
| Registros **com** outlier | **45,7%** |
| Registros **sem** outlier | 15,6% |
| Diferença | **+30,1 pp** |

Se fossem ruído de medição, sua taxa de evasão seria semelhante à do resto. É **três vezes maior**.
Remover ou winsorizar eliminaria preferencialmente os alunos em risco — a própria população-alvo.
**Nenhum registro foi removido.**

### 3.6 Multicolinearidade

Quatro pares com |r| ≥ 0,80, todos entre a mesma variável em semestres consecutivos — o maior em 0,945
(disciplinas creditadas). Sob essa magnitude, os coeficientes da Regressão Logística ficam instáveis: o
modelo prediz bem, mas a leitura de "quanto cada variável pesa" perde confiabilidade. **É uma das razões
pelas quais a interpretabilidade final vem do SHAP, não dos coeficientes.**

---

## 4. A falha de integridade encontrada no banco

Este é o achado metodológico mais relevante da Fase 2, e está separado porque tem valor próprio.

### 4.1 O que havia de errado

Duas colunas de nota estavam gravadas como `TEXT` no MySQL, com o ponto decimal reinterpretado como
separador de milhar na importação do CSV:

| Valor real | Como estava gravado |
|---|---|
| `13,42857142857140` | `"13.428.571.428.571.400"` |
| `11,4375` | `"114.375"` |
| `13,30625` | `"1.330.625"` |

Lidas com um `float()` ingênuo, produziriam **notas de 114 numa escala de 0 a 20** — contaminando
médias, correlações, o `StandardScaler` e as duas variáveis mais preditivas do dataset. Afetava **3.230
valores** (≈36% de cada coluna).

Havia também um **BOM** corrompendo o nome da primeira coluna (`ï»¿Marital status` → `imarital_status`).

### 4.2 Por que passou por todas as verificações padrão

| Verificação | Detectou? |
|---|---|
| `df.isnull().sum()` | ❌ |
| `df.duplicated().sum()` | ❌ |
| Colunas constantes | ❌ |
| `df.describe()` | ❌ |
| **Tipo declarado no schema × tipo esperado** | ✅ |
| **Faixa observada × domínio da variável (0–20)** | ✅ |

### 4.3 Por que a correção é determinística

Toda nota não nula do dataset está entre **9,80 e 18,875** — sempre com exatamente dois dígitos na parte
inteira; as notas zero foram gravadas como `"0.0"`. Logo, a posição da vírgula decimal é **única**, não
um chute.

**Auditoria:** a regra foi conferida contra a fonte oficial (UCI id=697) linha a linha —
**37 de 37 colunas idênticas, zero divergências em 4.424 linhas**. A auditoria é uma célula
reexecutável do Notebook 1.

### 4.4 As três lições

1. **Ausência de nulos não é qualidade.** As três verificações que todo tutorial de EDA ensina passaram
   sem alarme. O erro apareceu ao confrontar o **tipo declarado** com o esperado e a **faixa observada**
   com o domínio da variável.
2. **Uma correção precisa ser justificável, não conveniente.** Esta só é defensável porque o domínio
   torna a posição do separador única, e porque foi verificada contra o original. Registrar esse
   raciocínio é o que distingue **tratamento** de **manipulação** de dado.
3. **Toda correção precisa de validação posterior.** A primeira versão da regra estava incompleta e
   deixaria 39 valores errados passarem. Foi a validação de faixa — não a inspeção visual — que
   encontrou a falha.

> **Recomendação pendente para o banco:** as colunas continuam `TEXT` e deformadas no MySQL. O pipeline
> corrige em memória, mas qualquer consulta fora dele lerá valores errados. O SQL de reparo está em
> [`docs/02`](docs/02-qualidade-e-integridade.md), §6 — **não foi executado**, por ser operação
> destrutiva.

---

## 5. Fase 3 — Preparação dos dados

### 5.1 O princípio organizador

> **Nada que dependa dos dados pode ser aprendido antes da separação treino/teste.**

Média e desvio do `StandardScaler`, categorias do `OneHotEncoder` e vizinhos sintéticos do SMOTE são
**parâmetros estimados dos dados**. Calculá-los sobre o conjunto completo transmite informação do teste
ao modelo — e **o erro é silencioso, porque produz números melhores, não piores**.

### 5.2 A decisão sobre `Enrolled`

**Formulação binária adotada** (`Dropout` × `Graduate`), com `Enrolled` removido.

`Enrolled` não é desfecho — é **ausência de desfecho**. Em análise de sobrevivência isso se chama
**censura à direita**. Tratá-lo como terceira categoria equivaleria a afirmar que "continuar
matriculado" é um resultado final.

A formulação multiclasse está implementada e ativável em uma linha, para reportar como análise de
sensibilidade.

> **Limitação decorrente — a mais séria do trabalho.** O modelo passa a ser treinado apenas em
> **trajetórias concluídas**, mas seria aplicado a **alunos ativos** — precisamente a população excluída
> (794 alunos, 17,9%).

### 5.3 Transformação da dimensionalidade

| Etapa | Registros | Colunas |
|---|---:|---:|
| Base auditada | 4.424 | 37 |
| Após remover `Enrolled` | 3.630 | 37 |
| Treino (80%) | 2.904 | 36 |
| Teste (20%) | 726 | 36 |
| Após codificação e padronização | — | **105** |
| Treino após SMOTE | **3.534** | 105 |

**105 features** = 19 contínuas padronizadas + 8 binárias + 78 dummies. As nove nominais somavam **200
categorias**, 131 delas presentes em menos de 1% do treino; `min_frequency=0,01` agrupou as raras **sem
descartar nenhuma variável**.

### 5.4 SMOTE, e por que só no treino

O treino tinha 1.137 evasões contra 1.767 conclusões. O SMOTE criou **630 alunos sintéticos** da classe
minoritária, levando a razão de 1,55:1 para 1,00:1.

> Um ponto sintético é interpolado a partir de alunos reais. Se ele for para o treino enquanto seus
> "pais" ficam no teste, o modelo terá visto uma **versão borrada dos registros de teste**. A métrica
> resultante mediria memorização, não generalização.

**O conjunto de teste permaneceu com 726 registros e 39,1% de evasão** — a proporção natural. Nenhuma
métrica do trabalho foi calculada sobre dados sintéticos.

### 5.5 Nenhuma seleção de variáveis

Três razões: (a) interpretabilidade exige o quadro completo — uma variável irrelevante no agregado pode
ser decisiva para um subgrupo; (b) não há restrição computacional; (c) seleção agressiva sem
justificativa é decisão arbitrária.

A Fase 5 **justificou essa decisão por evidência** — ver §7.3.

### 5.6 Verificações que abortam a execução

```
[OK] treino balanceado (razão = 1,00)
[OK] teste NÃO balanceado (proporção natural)   <- falharia se o SMOTE contaminasse o teste
[OK] mesmas features em treino e teste
[OK] sem valores ausentes nem infinitos
```

Prova indireta do ajuste só no treino: após a padronização, o treino tem média **+0,000000** e desvio
**1,000000**, mas o **teste tem média −0,022141 e desvio 0,981818**. Se o scaler tivesse visto o
conjunto completo, o teste também estaria centrado.

---

## 6. Fase 4 — Modelagem

### 6.1 Protocolo

| Item | Definição |
|---|---|
| Validação | `StratifiedKFold(5, shuffle=True, random_state=42)` |
| SMOTE | Dentro do `imblearn.Pipeline` — **reaplicado por dobra**, nunca na validação |
| Busca de hiperparâmetros | `GridSearchCV` exaustivo — 540 (RF) + 1.215 (XGBoost) = **1.755 ajustes**; com a validação cruzada da linha de base, **1.760 no total** |
| Métrica de decisão | F1 da classe `Dropout`; recall e PR-AUC como desempates |
| Avaliação final | 726 alunos reais, 39,1% de evasão, sem dado sintético |

**Por que o SMOTE dentro do pipeline:** com o SMOTE aplicado ao treino inteiro, pontos sintéticos
derivados de um aluno da dobra de **validação** apareceriam nas dobras de **treino**. O
`imblearn.Pipeline` aplica reamostragem só no `fit`, nunca no `predict`.

### 6.2 O piso trivial

| Estratégia | Acurácia | Recall | F1 |
|---|---:|---:|---:|
| Sempre `Graduate` | **0,6088** | 0,0000 | 0,0000 |
| Sempre `Dropout` | 0,3912 | **1,0000** | 0,5624 |
| Aleatório 50/50 | 0,4931 | 0,5246 | 0,4474 |

"Sempre `Graduate`" tem 60,9% de acurácia e recall **zero** — inútil, e a acurácia não denunciaria.
"Sempre `Dropout`" tem recall **perfeito** sinalizando todo mundo — razão pela qual recall isolado
também não serve como critério único.

### 6.3 Resultados no conjunto de teste

| Modelo | Acurácia | Precisão | Recall | F1 | ROC-AUC | PR-AUC | MCC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Regressão Logística | 0,9132 | 0,8576 | **0,9331** | 0,8938 | 0,9739 | 0,9710 | 0,8227 |
| Random Forest | 0,9270 | 0,9140 | 0,8979 | 0,9059 | 0,9726 | 0,9697 | 0,8463 |
| **XGBoost** ✅ | 0,9270 | 0,9140 | 0,8979 | **0,9059** | **0,9747** | **0,9741** | 0,8463 |

**Hiperparâmetros do XGBoost:** `n_estimators=200`, `max_depth=3`, `learning_rate=0.1`,
`subsample=1.0`, `colsample_bytree=0.8`.

### 6.4 O modelo final em linguagem operacional

| | n | Significado |
|---|---:|---|
| Verdadeiros positivos | 255 | Evadidos corretamente sinalizados |
| Falsos negativos | **29** | Evadidos não detectados — **o erro caro** |
| Falsos positivos | 24 | Concluintes sinalizados por engano |
| Verdadeiros negativos | 418 | Concluintes corretamente liberados |

> **De cada 100 alunos que evadem, o modelo identifica 90 nesta partição dos dados. De cada 100
> sinalizados, 91 realmente evadiriam.** Total convocado: 279 de 726 — 38,4% da turma.

> **Ressalva de partição.** A [Verificação 2](#10-verificação-2--significância-estatística) mostrou que a
> semente 42 produziu uma partição favorável. Os valores típicos entre dez partições:
>
> | | Semente 42 | Típico | Faixa |
> |---|---:|---:|---|
> | Detectados de cada 100 evadidos | **90** | **84** | 82 – 86 |
> | Acerto de cada 100 sinalizados | 91 | 92 | 89 – 94 |
> | Falsos negativos | **29** | **44** | 40 – 50 |
> | Alunos sinalizados | 279 | 261 | 253 – 275 |
>
> A precisão **não apresentou** o mesmo viés otimista do recall — a semente 42 fica inclusive um pouco
> abaixo da média. Isso é um achado desta amostra, não uma propriedade estrutural do modelo.

### 6.5 O empate entre Random Forest e XGBoost

Os dois produzem **matriz de confusão idêntica** (418/24/29/255 na partição da semente 42) e ainda
assim **discordam sobre 24 alunos** — e nesses 24 cada modelo acerta exatamente **12**, o que torna a
compensação literal. As probabilidades atribuídas diferem (correlação 0,9712), confirmando que são
modelos distintos com o mesmo placar, não erro de transcrição.

| Par | Alunos com classificação divergente |
|---|---:|
| Reg. Logística × Random Forest | 54 (7,4%) |
| Reg. Logística × XGBoost | 52 (7,2%) |
| Random Forest × XGBoost | 24 (3,3%) |
| **Concordância dos três** | **661 (91,0%)** |

> Isso ilustra um limite das métricas agregadas: elas resumem o placar, não a decisão individual. Quando
> o produto final é uma lista de nomes para a coordenação abordar, **os 24 alunos divergentes são
> justamente o que importa** — argumento direto a favor da análise individual da Fase 5.

Sem um terceiro critério declarado, o vencedor sairia da ordem de iteração do dicionário. Aplicou-se
**PR-AUC**, que avalia o ordenamento de probabilidades em toda a faixa de limiares.

### 6.6 O trade-off que o F1 esconde

| Partição da semente 42 | Reg. Logística | XGBoost |
|---|---:|---:|
| Evadidos detectados | **265** de 284 | 255 de 284 |
| Evadidos perdidos | **19** | 29 |
| Falsos alarmes | 44 | **24** |
| Alunos convocados | 309 (42,6%) | **279 (38,4%)** |
| F1 | 0,8938 | **0,9059** |

A Regressão Logística encontra **10 evadidos a mais** ao custo de **20 convocações desnecessárias a
mais**. A [Verificação 2](#10-verificação-2--significância-estatística) mostrou que esse trade-off é
**estatisticamente significativo** (McNemar restrito aos evadidos, p = 0,034).

### A contradição — e como ela se resolve

Isso cria uma contradição a ser enfrentada, não suavizada: **o critério de seleção (F1) e a premissa de
custo assimétrico (recall > precisão) apontam para modelos diferentes.** Existe evidência estatística de
que o modelo escolhido é pior na dimensão que o trabalho declara ser a mais importante.

**O erro estava em tratar duas decisões como uma.** Comparar os dois modelos cada um no seu limiar padrão
de 0,50 compara **pontos de operação diferentes** — a Regressão Logística opera em recall 0,933 e o
XGBoost em 0,898. A comparação correta iguala o ponto de operação.

Calculando a **curva de custo exata** de cada modelo (ordenando os alunos pelo escore e obtendo, para
cada número de evadidos detectados, o mínimo de falsos alarmes necessário — sem interpolação, com
resolução de um aluno):

| Região | Recall | Pontos | Melhor modelo | Convocados |
|---|---|---:|---|---|
| **Estável** | 0,852 – 0,9577 | 31 | **XGBoost em 31 de 31** | 34% a 46% da turma |
| **Banda estreita** | 0,9613 – 0,9718 | 4 | **Reg. Logística em 4 de 4** | 47% a 49% da turma |
| **Instável** | ≥ 0,9754 | 8 | oscila — 3 travessias de liderança | 59% a 94% da turma |

*Convocação considerando os dois modelos; pelo XGBoost isoladamente, 34%–45%, 47%–49% e 59%–91%.*

**A manchete:** na faixa de recall 0,852 a 0,9577 o XGBoost vence em **31 de 31** pontos de operação,
convocando de 34% a 46% da turma.

**Com a ressalva que ela exige:** os quatro pontos em que a Regressão Logística vence convocam de
47% a 49% — **abaixo do teto prático de 50%** adotado neste trabalho. Não estão fora da faixa
utilizável; são os últimos quatro pontos antes da fronteira. Contabilizando tudo abaixo do teto, o
XGBoost vence **72 dos 76 pontos decididos**, e a Regressão Logística os outros 4. Adotar 62% em vez
de 50% como fronteira levaria a 74 × 4 — as mesmas quatro inversões, porque o segundo bloco só
aparece a partir de 66% de convocação.

**Mas há uma banda estreita real**, e esconder isso seria pior do que mostrá-la: entre recall 0,9613 e
0,9718 — quatro pontos consecutivos — a Regressão Logística é consistentemente melhor, por 1 a 11 falsos
alarmes, num regime que exige convocar praticamente metade da turma.

No **isocusto** a vantagem do XGBoost é marginal e oscila:

| Falsos alarmes | XGBoost detecta | Reg. Logística detecta | Diferença |
|---:|---:|---:|---:|
| 24 | 255 | 254 | +1 |
| 35 | 266 | 261 | +5 |
| 40 | 267 | 265 | +2 |
| 44 | 267 | 266 | +1 |
| 58 | 272 | 269 | +3 |

### O achado que dispensa comparação entre modelos

No limiar 0,50 a Regressão Logística detecta 265 evadidos pagando **44** falsos alarmes. Na sua própria
curva, os mesmos 265 custam apenas **40**.

> **O limiar padrão desperdiça 4 convocações do próprio modelo, sem ganho em detecção.** Isso demonstra,
> sem comparar um modelo com o outro, que **0,50 é convenção de biblioteca e não ponto de operação de
> nenhum dos dois** — a evidência mais econômica de que o limiar importa mais que o algoritmo.

### A hierarquia correta de decisão

| Ordem | Decisão | Quem decide |
|---:|---|---|
| 1º | **Política de custo** — quantas convocações desnecessárias valem a pena para não perder um aluno? | A instituição |
| 2º | **Limiar de decisão** que implementa essa política | Derivado de 1 |
| 3º | **Algoritmo** — só relevante se a política exigir recall acima de 0,96 | Técnico |

O trabalho vinha invertendo essa ordem. **A escolha do algoritmo é a menos importante das três.**

**Por que F1 permanece o critério de seleção:** comparabilidade com a literatura desta base, e porque a
magnitude absoluta da diferença em qualquer ponto pareado é de +1 a +5 evadidos — menor que a variação
induzida pela partição dos dados.

**A Regressão Logística permanece alternativa legítima**, por simplicidade e transparência, por não haver
perda relevante de desempenho (ICs de *bootstrap* sobrepostos), e pela vantagem real no regime de recall
muito alto — se a instituição tiver capacidade de convocar metade da turma.

> **Ressalva metodológica:** qualquer limiar localizado por busca no conjunto de teste é ilustrativo do
> formato da curva, não um valor a adotar em produção sem validação em dados novos.

### 6.7 Calibração do limiar de decisão

O limiar de 0,50 é convenção de biblioteca, não decisão pedagógica.

| Limiar | Sinalizados | Detectados | **Não detectados** | Falsos alarmes |
|---:|---:|---:|---:|---:|
| 0,10 | 447 | 278 | **6** | 169 |
| 0,20 | 366 | 276 | **8** | 90 |
| **0,30** | 330 | 272 | **12** | 58 |
| **0,50** ⬅ adotado | **279** | **255** | **29** | **24** |
| 0,64 (F1 máximo) | 260 | 250 | 34 | 10 |
| 0,80 | 240 | 237 | 47 | 3 |

- Baixar de 0,50 para **0,30** reduz os evadidos não detectados de 29 para 12 — **17 alunos resgatados**
  por 34 conversas extras.
- **Elevar** o limiar é mau negócio: de 0,50 para 0,80 economizam-se 21 conversas ao custo de 18 alunos
  perdidos.

Manteve-se 0,50 nas métricas reportadas: otimizar o limiar *no conjunto de teste* seria sobreajuste
sutil ao próprio teste.

### 6.8 A conclusão central da fase

O ganho do XGBoost sobre a linha de base linear é de **+1,4% em F1**. Com 726 registros de teste, cada
aluno reclassificado move o F1 em ~0,0019 — uma diferença de 0,01 equivale a **5 alunos**.

> **O sinal preditivo está nos dados, não no algoritmo.** Como a complexidade rende pouco, **a
> interpretabilidade pode ser tratada como critério legítimo de decisão, não como concessão** — o que
> sustenta diretamente o objetivo do trabalho.

---

## 7. Fase 5 — Interpretabilidade

### 7.1 Fundamento e verificação

`shap.TreeExplainer` (**TreeSHAP**) calcula os valores de Shapley **exatos** em tempo polinomial.
Foram calculadas **76.230 contribuições** (726 alunos × 105 features).

A propriedade central foi verificada numericamente:

```
predição(aluno) = valor base + Σ SHAP_i(aluno)
erro máximo em 726 alunos: 3,8 × 10⁻⁶
```

O resíduo é da ordem da precisão de `float32`. **A explicação é exata, não aproximada** — e auditável:
qualquer valor pode ser somado de volta à predição.

### 7.2 Importância e direção

Direção medida pela correlação de Spearman entre o valor da variável e seu valor SHAP.

| # | Variável | % do impacto | ρ | Efeito de valor alto |
|---:|---|---:|---:|---|
| 1 | Disciplinas aprovadas (2º sem.) | **23,4** | −0,967 | reduz o risco |
| 2 | Disciplinas aprovadas (1º sem.) | 12,5 | −0,956 | reduz |
| 3 | Mensalidades em dia | 6,4 | −0,604 | reduz |
| 4 | Disciplinas **matriculadas** (1º sem.) | 5,9 | **+0,718** | **aumenta** |
| 5 | Média das notas (2º sem.) | 3,8 | −0,894 | reduz |
| 6 | Bolsista | 3,7 | −0,739 | reduz |
| 7 | Nota de admissão | 3,6 | +0,254 | não monotônico |
| 8 | Idade no ingresso | 3,1 | +0,786 | aumenta |
| 9 | Taxa de desemprego | 2,7 | **+0,943** | **aumenta** |
| 10 | Inadimplente | 2,7 | +0,554 | aumenta |

Concentração: top 3 = **42,3%**; top 10 = **67,8%**. **24 das 105 features têm impacto nulo.**

### Por grupo conceitual

| Grupo | Features | % do impacto |
|---|---:|---:|
| **Desempenho acadêmico** | 12 | **59,9** |
| **Situação financeira** | 3 | **14,9** |
| Trajetória de ingresso | 36 | 9,5 |
| Perfil demográfico | 11 | 5,7 |
| Contexto macroeconômico | 3 | 5,1 |
| **Origem familiar** | **40** | **4,9** |

O contraste nos extremos é o mais informativo: **desempenho acadêmico entrega 59,9% do impacto com 12
features**, enquanto **origem familiar entrega 4,9% com 40** — a maior contagem e o menor efeito.
Provavelmente porque o capital cultural atua **mediado** pelo desempenho, que já está no modelo.

### 7.3 O achado principal: efeito condicional contra efeito marginal

**Três variáveis receberam do SHAP uma direção oposta à que a análise descritiva indicava.**

| Variável | Efeito marginal | Efeito condicional | Situação |
|---|---|---|---|
| Deslocado da cidade | r = −0,126 — **protege** (33,6% × 45,9%) | ρ = **+0,861** | inversão de sinal |
| Disciplinas matriculadas (1º sem.) | r = −0,161 — protege | ρ = **+0,718**, 4ª mais importante | inversão de sinal |
| Taxa de desemprego | r = **−0,004** — nulo | ρ = **+0,943**, 9ª mais importante | efeito que **emerge** |

**O caso das matrículas** tem leitura direta: *dado* o número de disciplinas aprovadas, matricular-se em
muitas passa a sinalizar **sobrecarga e reprovação acumulada**. Matricular 8 e aprovar 6 é uma coisa;
matricular 8 e aprovar 1 é outra.

**O caso do desemprego** é o mais notável: a relação marginal tem **forma de U** (41,0% → 38,2% → 35,3%
→ 43,2%), então a correlação linear a mede como nula **por cancelamento**. Condicionada ao perfil
individual, é substancial — e isso **confirma a literatura**, ao contrário do que a Fase 2 supunha.

> **Por que isto importa.** A análise descritiva isolada levaria a conclusões erradas sobre as três —
> inclusive à tentação de descartar a taxa de desemprego por correlação nula. **A decisão de não fazer
> seleção de variáveis passou a ter justificativa empírica, não apenas de princípio.**
>
> Em um modelo com 105 variáveis correlacionadas, o efeito marginal de uma variável isolada e sua
> contribuição dentro do modelo são **grandezas diferentes**. Confundi-las é o erro mais comum na
> leitura de resultados de machine learning aplicado.

### 7.4 Recomposição das variáveis categóricas

A linearidade dos valores de Shapley autoriza somar as dummies de uma mesma variável original — operação
exata, não aproximada.

| Variável | Dummies | Melhor dummy | Recomposta | Ganho |
|---|---:|---:|---:|---:|
| Ocupação do pai | 13 | 0,0466 | 0,1174 | **2,5×** |
| Curso | 17 | 0,0965 | 0,2332 | **2,4×** |
| Escolaridade da mãe | 8 | 0,0303 | 0,0603 | 2,0× |

Após a recomposição, 105 features → 35 variáveis originais, e `Curso` sobe para a **8ª posição**. **Este
é o ranking correto** quando a pergunta é "quais fatores importam?" — o da §7.2 responde a outra
pergunta, "quais *features do modelo* importam?".

### 7.5 Gatilhos operacionais

Convertidos de volta pela inversão do `StandardScaler` — "valor padronizado abaixo de −0,175" não é
acionável por uma secretaria; "até 3 aprovações no semestre" é.

| Variável | Gatilho de alerta | SHAP no pior valor |
|---|---|---:|
| **Disciplinas aprovadas (2º sem.)** | até **3** disciplinas | +2,43 (com 0 aprovações) |
| Disciplinas aprovadas (1º sem.) | até **4** disciplinas | +1,04 |
| Média das notas (2º sem.) | até **11** (escala 0–20) | +0,36 |
| Idade no ingresso | **23** anos ou mais | +0,51 |

**A assimetria é informativa:** estar inadimplente contribui com **+1,86**; estar em dia, com apenas
**−0,19**. O modelo aprendeu que *ir mal antecipa a evasão, mas ir bem não garante a formatura*.

### 7.6 Convergência entre os três métodos

| Par de métodos | ρ de Spearman |
|---|---:|
| Correlação linear (NB1) × impureza (NB2) | +0,753 |
| Impureza (NB2) × SHAP (NB4) | +0,774 |
| Correlação linear (NB1) × SHAP (NB4) | +0,697 |

**O que converge:** o bloco dominante é o mesmo nos três métodos. Essa é a conclusão substantiva, e ela
sobrevive a três operacionalizações com fundamentos matemáticos distintos.

**O que não converge:** um ρ de 0,70 a 0,77 está longe de 1,00 — há reordenação substancial nas posições
individuais. E isso é esperado, dado o achado da §7.3.

> **Consequência para a redação:** afirmações sobre **blocos** ("desempenho acadêmico é o fator
> dominante") são robustas. Afirmações sobre **posições específicas** ("a variável X é a 4ª mais
> importante") são condicionais ao método.

### 7.7 O falso negativo — o teto intrínseco

**O caso C é típico do grupo?** Descrever um aluno e concluir que existe teto intrínseco é inferência
a partir de n = 1. Testado no agregado:

| Variável | Falsos negativos | Evadidos detectados | Formandos |
|---|---:|---:|---:|
| Disciplinas aprovadas (2º sem.) | **6,24** | 1,23 | 6,10 |
| Disciplinas aprovadas (1º sem.) | **6,76** | 1,82 | 6,12 |
| Média das notas (2º sem.) | **12,37** | 5,00 | 12,55 |
| Mensalidades em dia | **0,97** | 0,62 | 0,99 |
| Idade no ingresso | 24,55 | 26,39 | 21,60 |
| Bolsista | 0,07 | 0,09 | 0,33 |

Em **5 das 7** variáveis-chave o falso negativo médio está mais próximo do formando. O caso C é retrato
do grupo, não exceção. Duas variáveis fogem — são menos bolsistas e mais velhos —, ou seja: parecem
formandos no desempenho e nas finanças, mas mantêm marcadores demográficos de risco.

> **Núcleo duro, verificado.** Entre os 493 evadidos que apareceram no conjunto de teste ao menos 3
> vezes nas 10 partições, **68 (13,8%) foram classificados errado em 100% das vezes**. Erro aleatório
> à taxa global de 15,6% produziria 1,3 alunos nessa condição — observam-se 68, cerca de 54 vezes
> mais. A distribuição é bimodal (80,7% nunca errados, 13,8% sempre errados, menos de 6% no meio). O
> erro do modelo não é ruído distribuído: concentra-se num subconjunto identificável. É esse núcleo,
> e não a variação amostral, que sustenta a afirmação de teto intrínseco.

Um aluno que **evadiu** e recebeu escore de apenas **0,023**:

| SHAP | Fator |
|---:|---|
| −1,610 | Disciplinas aprovadas (2º sem.) |
| −1,185 | Disciplinas aprovadas (1º sem.) |
| +0,436 | Disciplinas matriculadas (1º sem.) |

O perfil é **dominado por fatores protetores**. Nos indicadores que o modelo usa, o aluno parecia
regular.

> Existe um **teto que não depende de algoritmo, hiperparâmetro ou volume de dados**: as causas de parte
> das evasões **não estão nas 36 variáveis disponíveis** — saúde, mudança de cidade, questões
> familiares, oferta de emprego. Nenhum ajuste de modelagem recuperaria esses 29 alunos.
>
> Consequência prática direta: o sistema deve ser apresentado como **auxílio à priorização, nunca como
> veredito**. **Um aluno não sinalizado não é um aluno seguro.**

### 7.8 Confronto com a literatura

| Achado | Literatura | Situação |
|---|---|---|
| Desempenho acadêmico inicial domina (59,9%) | Tinto (1975, 1993) — integração acadêmica | ✅ Confirma |
| Situação financeira forte (14,9%) | Cabrera et al. (1992); Realinho et al. (2022) | ✅ Confirma |
| Bolsa protege (ρ = −0,74) | Literatura sobre auxílio estudantil | ✅ Confirma |
| Idade maior aumenta o risco (ρ = +0,79) | Estudante não tradicional | ✅ Confirma |
| **Desemprego aumenta o risco** (9ª mais importante) | Associação desemprego–evasão | ✅ Confirma — **só no condicional** |
| Origem familiar modesta (4,9% com 40 features) | Capital cultural como determinante forte | ⚠️ Nuance — efeito **mediado** |

### 7.9 Recomendações para a gestão

| # | Recomendação | Custo | Ressalva causal |
|---|---|---|---|
| **R-1** | Monitorar aprovações por semestre (gatilho: até 3 no 2º sem.) | baixo | Indicador **tardio** — quando dispara, o aluno já parou |
| **R-2** | Tratar inadimplência como alerta acadêmico, não só financeiro | baixo | Pode ser **sintoma** da decisão de sair, não causa |
| **R-3** | Ampliar e priorizar bolsas | alto | **Viés de seleção provável** — bolsas exigem desempenho mínimo |
| **R-4** | Acompanhamento para ingressantes de 23+ anos | médio | Idade é **marcador**, não causa |
| **R-5** | **Encaminhar ao setor conforme o fator dominante** | baixo | Ver revisão na [Verificação 3](#11-verificação-3--equidade-algorítmica) |
| **R-6** | Comunicar como priorização, nunca veredito | nulo | Os 29 falsos negativos desta partição — 44 numa típica — exigem isso |
| **R-7** | Calibrar o limiar conforme a capacidade da equipe | nulo | Decisão de política, não de modelagem |

**R-5 em detalhe** — a recomendação que só o SHAP viabiliza:

| Fator dominante no SHAP | Encaminhamento |
|---|---|
| Inadimplência, ausência de bolsa | Assistência estudantil / setor financeiro |
| Reprovações, notas baixas | Apoio pedagógico / monitoria / tutoria |
| Poucas avaliações realizadas | Contato direto — possível abandono em curso |
| Idade, turno noturno | Orientação sobre flexibilização de carga |

> **A advertência que atravessa todas as recomendações: SHAP explica o modelo, não o mundo.** Os valores
> medem **como o XGBoost usa cada variável**, não o efeito causal sobre a evasão. As recomendações
> identificam **onde intervir**; verificar **se a intervenção funciona** exigiria delineamento
> experimental.

---

## 8. Fase 6 — Consolidação e ferramenta de uso

O Notebook 5 consolida os resultados e acrescenta o único elemento inédito da fase: uma
**demonstração de uso ponta a ponta**.

A função `avaliar_aluno()` recebe os dados **brutos** de um aluno — como sairiam do sistema acadêmico,
sem padronização nem codificação — e devolve escore de risco, explicação individual e **encaminhamento
sugerido**, usando apenas `preprocessador.joblib` + `melhor_modelo.joblib`.

Para viabilizá-la, o notebook **reconstrói a partição** treino/teste a partir do `dataset_limpo.csv` e
verifica que reproduz exatamente o `X_test.csv` do Notebook 2 — o que também é prova de
reprodutibilidade do pipeline.

Aplicada a quatro alunos reais. O caso mais informativo foi o de **risco moderado** (escore 0,545, real
`Graduate`): os fatores decisivos foram **3 aprovações no 2º semestre e 4 no 1º** — exatamente os
gatilhos identificados na Fase 5. Os limiares derivados do SHAP e o comportamento do modelo em um caso
concreto coincidem.

---

## 9. Verificação 1 — Calibração de probabilidade

**Motivação:** a análise de limiar trata a saída do XGBoost como probabilidade real. Um modelo treinado
com SMOTE aprende a distribuição artificial 50/50 — e o valor base do SHAP ficou em **0,5025** contra a
taxa real de 39,1%.

### 9.1 O diagnóstico

| Medida | Valor |
|---|---|
| Brier score | 0,0520 |
| ECE (10 faixas) | 0,1052 |
| Probabilidade média predita | 0,4284 |
| Taxa real de evasão | 0,3912 — IC 95% [0,3563 — 0,4272] |
| **Diferença** | **+3,7 pp — fora do IC** |

**O modelo superestima sistematicamente o risco médio.** A causa é conhecida: é o efeito esperado do
SMOTE.

### 9.2 A distribuição é bimodal

**Dois terços dos alunos (68,6%) caem nas faixas extremas** — e ali a calibração é praticamente perfeita
(0,043 × 0,022 e 0,981 × 1,000). O modelo é **decidido**, coerente com as fronteiras em degrau que a
Fase 5 mostrou.

Das três faixas com miscalibração detectável, **apenas uma importa**:

| Faixa | n | Predito × observado | Natureza |
|---|---:|---|---|
| [0,9 – 1,0) | 219 | 0,981 × 1,000 | desvio de 2 pp — irrelevante |
| [0,5 – 0,6) | 14 | 0,554 × 0,214 | **ruído amostral** |
| **[0,1 – 0,2)** | **81** | **0,142 × 0,025** | **miscalibração real** |

Nessa faixa, **81 alunos (11% do teste)** recebem ~14% de risco quando a frequência real é 2,5% — o
modelo é **excessivamente cauteloso**. É uma razão adicional para não baixar o limiar agressivamente: um
corte em 0,10 convocaria esses 81 alunos, dos quais apenas 2 evadiriam.

### 9.3 A calibração posterior não foi adotada

| Método | Brier | ECE | ROC-AUC | F1 @0,50 |
|---|---:|---:|---:|---:|
| **sem calibração** | 0,0520 | 0,1052 | 0,9747 | **0,9059** |
| sigmoid | 0,0523 | 0,1189 | 0,9747 | 0,9046 |
| isotonic | 0,0516 | 0,0880 | 0,9747 | 0,9071 |

O isotonic reduz o **ECE em 16%** (0,1052 → 0,0880), mas ganha apenas **0,0004 de Brier**, e não
corrige a faixa [0,10–0,20) que motivou a análise: o erro ali cai de +11,8 pp para +10,7 pp. O sigmoid
**piora** as duas métricas agregadas. A razão decisiva, porém, é outra: a calibração é **monotônica**,
o ROC-AUC é idêntico nos três casos, e portanto ela não altera quem seria convocado. A razão de
fundo: a miscalibração está na faixa central pouco populada, onde nenhum calibrador tem dados para
corrigir de forma confiável. **`melhor_modelo.joblib` permanece inalterado.**

O **ROC-AUC é idêntico nos três casos** — calibração é transformação monotônica, portanto **não altera a
ordenação dos alunos**.

### 9.4 Veredito

**A análise de limiar permanece válida como está.** Ela não é inferência probabilística: varre o limiar e
**conta alunos reais**. Depende apenas da ordenação, que a calibração não altera.

Confirmação empírica: entre os escores 0,30 e 0,50 há **51 alunos, dos quais 17 evadiram e 34 não** —
exatamente o que a Fase 4 afirmava ("17 alunos resgatados ao custo de 34 conversas").

**O que muda é a linguagem:**

| Evitar | Usar |
|---|---|
| "probabilidade de evasão de 85%" | "**escore de risco** de 0,85" |
| "30% dos alunos nesta faixa evadem" | "ponto de corte 0,30 na ordenação" |

Nenhum resultado enfraquece — F1, recall e a análise de limiar dependem só da ordenação.

---

## 10. Verificação 2 — Significância estatística

**Motivação:** as métricas finais vêm de uma única partição de teste. A tese de equivalência entre
modelos é sustentável estatisticamente? E o resultado com `random_state=42` é representativo?

### 10.1 *Bootstrap* — 1.000 reamostragens

| Modelo | F1 | IC 95% |
|---|---:|---|
| Regressão Logística | 0,8938 | [0,8663 — 0,9168] |
| Random Forest | 0,9059 | [0,8786 — 0,9293] |
| XGBoost | 0,9059 | [0,8791 — 0,9300] |

**Nenhum par tem intervalos disjuntos**, em nenhuma das três métricas. A largura do IC em F1 é **0,051**
contra **0,012** de diferença entre o melhor e o pior modelo — a incerteza amostral é **quatro vezes
maior** que o efeito.

### 10.2 McNemar — e o achado que revisa uma conclusão

**Escopo global (acurácia):** nenhum par significativo.

| Par | Só A acerta | Só B acerta | p |
|---|---:|---:|---:|
| Reg. Logística × Random Forest | 22 | 32 | 0,2207 |
| Reg. Logística × XGBoost | 21 | 31 | 0,2120 |
| Random Forest × XGBoost | **12** | **12** | 0,8383 |

A simetria perfeita entre RF e XGBoost — 12 contra 12 — é a explicação estatística do empate exato nas
matrizes de confusão.

**Mas o teste global mede acurácia, que trata os dois erros como equivalentes** — exatamente o que o
trabalho nega. O McNemar **restrito aos 284 evadidos** testa o recall diretamente:

| Par | Só A detecta | Só B detecta | p | Significativo |
|---|---:|---:|---:|---|
| **Reg. Logística × Random Forest** | **15** | 5 | **0,0442** | **sim** |
| **Reg. Logística × XGBoost** | **14** | 4 | **0,0339** | **sim** |
| Random Forest × XGBoost | 5 | 5 | 0,7518 | não |

> **A tese central precisa de uma qualificação.** Os três modelos são estatisticamente equivalentes **em
> acurácia global e em F1**. Mas a **Regressão Logística detecta significativamente mais evadidos** — e
> o recall da classe `Dropout` é justamente a métrica que o trabalho declara priorizar.

**Ressalva de escopo — o teste é condicional ao ponto de operação.** O McNemar comparou os modelos **no
limiar padrão de 0,50 para ambos**. Como 0,50 é convenção de biblioteca, os dois modelos operam ali em
níveis de recall diferentes (0,933 e 0,898). O teste **não impede** que o recall seja igualado ajustando
o limiar de um deles — e a análise de custo pareado da §6.6 mostra que, igualados os pontos de operação,
**o XGBoost detecta mais evadidos pelo mesmo custo em 31 de 31 pontos da faixa de recall 0,852–0,9577**
— e em 72 dos 76 pontos decididos abaixo do teto de convocação de 50%.

**O que o teste estabelece:** que a diferença de recall **no limiar padrão** é real e não fruto do acaso.
**O que ele não estabelece:** que a Regressão Logística seja um modelo melhor. A diferença é uma
propriedade **do ponto de operação**, não do algoritmo.

### PR-AUC também não é significativo

O [doc 04](docs/04-modelagem.md) usou PR-AUC como critério de desempate. Submetido às mesmas 1.000
reamostragens:

| Modelo | PR-AUC | IC 95% |
|---|---:|---|
| Regressão Logística | 0,9710 | [0,9587 — 0,9816] |
| Random Forest | 0,9697 | [0,9567 — 0,9811] |
| XGBoost | 0,9741 | [0,9623 — 0,9846] |

Diferença **pareada** XGBoost − Reg. Logística: **+0,0031**, IC 95% **[−0,0027, +0,0093]** — **contém o
zero**. P(XGBoost > Reg. Logística) = 85,4% das reamostragens: sugestivo, não conclusivo.

> **O PR-AUC não sustenta a palavra "domina".** Isso não invalida seu uso como critério de desempate — um
> desempate precisa ser objetivo e declarado de antemão, não estatisticamente significativo. Mas onde o
> trabalho o usava como evidência de superioridade, a formulação correta é que ele é **consistente com**
> uma leve vantagem do XGBoost, sem estabelecê-la.

### 10.3 Dez sementes — o achado que não estava previsto

Primeiro, a validação: a reconstrução do pipeline reproduz **exatamente** os valores publicados com a
semente 42 (LR 0,8938 / RF 0,9059 / XGB 0,9059). A comparação é legítima.

| Modelo | Sementes 0–9 | Semente 42 | z | Supera |
|---|---|---:|---:|---:|
| Regressão Logística | 0,8755 ± 0,0085 | 0,8938 | +2,14 | **10/10** |
| Random Forest | 0,8735 ± 0,0086 | 0,9059 | +3,75 | **10/10** |
| XGBoost | 0,8796 ± 0,0099 | 0,9059 | +2,64 | **10/10** |

> **A semente 42 produziu uma partição favorável.** Supera todas as 10 alternativas testadas, nos três
> modelos. O F1 publicado do XGBoost fica **acima do máximo** observado entre as sementes 0–9 (0,8909).
>
> **O F1 esperado em uma partição típica é ≈ 0,88, não 0,906.**

**Não houve escolha oportunista.** `random_state=42` é a convenção padrão da área, foi fixada *a priori*
e nunca trocada em função de resultados. É uma **correção de expectativa, não de método** — e é exatamente
o que a limitação de "partição única" antecipava.

### 10.4 Estabilidade do ranking

| Métrica | Lidera em quantas sementes | Amplitude entre modelos | Desvio da partição |
|---|---|---:|---:|
| **F1** | XGBoost 6/10 · LR 3/10 · RF 1/10 | 0,0062 | 0,0090 |
| **Recall** | **Reg. Logística 10/10** | 0,0282 | 0,0128 |

- **Em F1, a amplitude entre modelos é menor que o ruído da partição** — mais evidência de equivalência.
- **Em recall, a amplitude é 2,2× o ruído e a Reg. Logística lidera em 10/10** — reproduzindo o McNemar
  por caminho independente.

> Quando a diferença entre dois modelos é menor que a variação induzida pela partição, a ordem entre eles
> é acidente de amostragem. **O que sobrevive à troca de semente é o que o trabalho pode afirmar.**

---

## 11. Verificação 3 — Equidade algorítmica

**Motivação:** a Fase 5 confirmou que o modelo usa gênero (ρ = +0,55) e variáveis socioeconômicas. Como
o produto é uma **lista de alunos que receberão apoio institucional**, um modelo que detecta pior um
subgrupo **amplifica a desigualdade que deveria combater**.

### 11.1 Resultados

| Variável | Menor recall | Maior recall | Disparidade | Disp. FPR | p | Significativo |
|---|---|---|---:|---:|---:|---|
| **Faixa etária** | até 20: 0,858 | 26+: 0,931 | **7,3 pp** | 2,4 pp | 0,0720 | não |
| Deslocado | deslocado: 0,878 | não desl.: 0,913 | 3,5 pp | 3,7 pp | 0,3433 | não |
| Turno | diurno: 0,894 | noturno: 0,923 | 2,9 pp | 0,0 pp | 0,5343 | não |
| Bolsista | não bolsista: 0,895 | bolsista: 0,923 | 2,8 pp | 4,1 pp | 0,6181 | não |
| Gênero | feminino: 0,885 | masculino: 0,909 | 2,4 pp | 4,0 pp | 0,5008 | não |

| Verificação | Resultado |
|---|---|
| Disparidades de recall acima de 10 pp | **nenhuma** |
| Disparidades estatisticamente significativas | **nenhuma** |
| Pares com ICs de recall disjuntos | **nenhum** |

### 11.2 A faixa etária é o ponto de monitoramento

A maior disparidade — **7,3 pp**, p = 0,072 — é marginal, não significativa. Dois detalhes a tornam
candidata a vigilância:

1. **A direção é monotônica:** o recall cresce com a faixa (0,858 → 0,909 → 0,931).
2. **Acompanha a taxa-base:** a evasão real também cresce (26,8% → 42,3% → 66,7%). O modelo detecta
   melhor onde a evasão é mais frequente — comportamento esperado, não viés dirigido.

Os alunos mais jovens são o grupo com maior risco de passar sem alerta. Como são também o mais numeroso
(422 de 726) e o de menor taxa de evasão, o efeito absoluto é limitado.

### 11.3 *Equalized odds*

Duas variáveis mostram FPR ≈ 2× diferente em termos relativos — gênero (0,042 × 0,082) e deslocamento
(0,031 × 0,068). Em termos **absolutos** é de 4 pp, e **os ICs se sobrepõem nos dois casos**.

> Para um sistema de **triagem para apoio**, um FPR mais alto significa *mais atenção oferecida*, não uma
> penalidade. A assimetria de custo aqui não é a de um sistema de crédito ou de justiça criminal.

### 11.4 A ressalva indispensável

| Subgrupo | Evadidos | Largura do IC de recall |
|---|---:|---:|
| Bolsista · bolsista | **26** | **22,0 pp** |
| Turno · noturno | 39 | 17,7 pp |
| Faixa etária · 21 a 25 anos | 55 | 15,6 pp |

> **Uma disparidade real de 10 a 20 pp em subgrupos pequenos seria indetectável com esta amostra.** A
> ausência de significância é **compatível** com equidade; não a comprova.
>
> Confundir "não detectei diferença" com "não há diferença" é o erro clássico de interpretação de teste
> de hipótese — e num contexto de alocação de apoio institucional, é um erro com consequências.

**A conclusão não é "o modelo é justo", e sim: não há evidência de injustiça nos subgrupos e no tamanho
de amostra examinados, e o monitoramento precisa continuar em operação.**

### 11.5 R-5 revisada

> **R-5 (revisada).** O uso do modelo para alocação de recursos deve ser acompanhado de **monitoramento
> contínuo de equidade**. Em operação real, com volume acumulado maior, as mesmas métricas devem ser
> recalculadas periodicamente — o `08_auditoria_equidade.ipynb` é reexecutável para esse fim.
>
> **Ponto de atenção prioritário:** o recall dos alunos mais jovens (até 20 anos) é o menor da amostra.

---

## 11b. Aplicação — o modelo sobre os 794 alunos ainda cursando

Documento detalhado: [`docs/13`](docs/13-aplicacao-alunos-ativos.md) · Notebook:
`09_aplicacao_alunos_ativos.ipynb`

O modelo foi aplicado à classe `Enrolled`, removida do treino na Fase 3. É o uso pretendido do trabalho
e o teste mais duro da limitação L-01. Esses alunos nunca passaram pelo modelo — foram excluídos antes
do split, sem vazamento.

### O achado central

| Grupo | n | Escore mediano | % na zona 0,2–0,8 |
|---|---:|---:|---:|
| Dropout — evadiram | 1.421 | 0,984 | 17,5% |
| **Enrolled — ainda cursando** | **794** | **0,480** | **42,4%** |
| Graduate — formaram | 2.209 | 0,058 | 13,1% |

Os concluídos ficam nos extremos; **os ativos ficam no meio**. Não é defeito do modelo: é consequência
direta de ele nunca ter visto essa população. A L-01 deixa de ser ressalva formal e passa a ter número.

**Consequência imediata:** o limiar 0,50 sinalizaria **386 alunos, 48,6% dos ativos** — não é política
de intervenção, é uma lista que ninguém atende. A saída é partir da capacidade real: com 50 vagas de
monitoria, convoque os 50 de maior escore (corte em 0,988).

### A entrega

Lista priorizada dos 794 alunos com os três fatores que mais pesaram em cada caso e o setor de
encaminhamento (`dados/alunos_ativos_priorizados.csv`). Distribuição por faixa: 181 muito alto, 105
alto, 150 moderado, 358 baixo. Entre os 100 de maior risco, 95 vão para o pedagógico e 5 para o
financeiro.

**442 alunos (55,7%) têm risco em mais de uma área** — são os que escapam de fluxos institucionais,
porque cada setor os trata como caso do outro.

### O limite

Nenhum número desta seção foi validado contra desfecho real, porque esses desfechos ainda não existem.
O escore responde a *"este aluno se parece com quem evadiu?"*, não a *"qual a probabilidade de ele
evadir"*. Serve para **priorizar atenção**, não para afirmar destino.

A validação real é registrar a lista hoje e conferir em dois semestres quem de fato evadiu — o que
converteria os 794 em conjunto de teste verdadeiro.

---

## 12. Conclusões consolidadas

### 12.1 O objetivo foi atingido?

O objetivo era **desenvolver um modelo computacional capaz de identificar alunos com risco de evasão**.
O resultado identifica **90 de cada 100 alunos que evadem nesta partição dos dados** — valor típico de
**84** entre dez partições, faixa de 82 a 86 —, com 91% de precisão entre os sinalizados (típico 92) e
convocando 38% da turma.

Mas o título pedia mais: um modelo **interpretável**. É aí que está a contribuição.

### 12.2 Sobre o fenômeno da evasão

1. **O risco se concentra em dois blocos que somam 74,8% do impacto:** desempenho acadêmico inicial
   (59,9%) e situação financeira (14,9%). O achado sobrevive a **três métodos independentes** —
   correlação linear, redução de impureza e valores de Shapley.
2. **A origem familiar tem efeito modesto** (4,9% com 40 das 105 features), provavelmente porque atua
   **mediada** pelo desempenho acadêmico, que já está no modelo.
3. **Os gatilhos são operacionalizáveis:** até 3 aprovações no 2º semestre, até 4 no 1º, média até 11,
   idade 23+.
4. **A assimetria risco/proteção é grande:** inadimplência contribui com +1,86; adimplência com −0,19.
   *Ir mal antecipa a evasão; ir bem não garante a formatura.*

### 12.3 Sobre a modelagem

5. **O ganho de um modelo complexo sobre a linha de base linear é de +1,4% em F1** — e o *bootstrap*
   confirma que **todos os intervalos de confiança se sobrepõem**. Como a complexidade rende pouco, **a
   interpretabilidade não custa desempenho**: deixa de ser concessão e passa a ser critério legítimo de
   escolha.
6. **Com uma qualificação, e ela é mais precisa do que parecia.** Pelo McNemar, a Regressão Logística
   detecta significativamente mais evadidos **no limiar padrão de 0,50** (p = 0,034). Mas o limiar 0,50 é
   convenção de biblioteca — e a curva de custo exata mostra que, **igualados os pontos de operação**, o
   XGBoost detecta mais evadidos pelo mesmo custo em **31 de 31** pontos na faixa de recall
   0,852–0,9577, e em **72 dos 76 pontos decididos** abaixo do teto de convocação de 50%. Os 4 pontos
   restantes pertencem ao modelo linear (recall 0,9613–0,9718) e convocam de 47% a 49% da turma —
   **dentro** do teto, nos últimos pontos percentuais antes dele. A diferença entre algoritmos é uma
   propriedade do **ponto de operação**, não dos modelos.

6b. **O limiar importa mais que o algoritmo — e há evidência que dispensa comparar modelos.** No limiar
   0,50 a Regressão Logística paga 44 falsos alarmes por 265 evadidos detectados, quando a sua própria
   curva alcança os mesmos 265 com 40. O limiar padrão desperdiça desempenho do próprio modelo. A
   hierarquia correta de decisão é: 1º a política de custo da instituição, 2º o limiar que a implementa,
   3º o algoritmo.
7. **O desempenho publicado é otimista, e a magnitude está quantificada.** A semente 42 supera 10/10
   alternativas; a expectativa típica é **F1 ≈ 0,88**, não 0,906 — e **84 detectados de cada 100
   evadidos**, não 90. A precisão não apresentou o mesmo viés.
8. **A saída é um escore de risco ordenável, não uma probabilidade calibrada.** O modelo superestima o
   risco médio em 3,7 pp — efeito do SMOTE. A análise de limiar não é afetada, porque conta alunos reais.

### 12.4 Sobre método

9. **Ausência de nulos não é qualidade.** As verificações padrão passaram sem alarme sobre 3.230 notas
   corrompidas. O erro apareceu ao confrontar **tipo declarado** e **domínio da variável**.
10. **Efeito marginal e efeito condicional são grandezas diferentes.** Três variáveis divergem entre os
    dois — a taxa de desemprego tem correlação marginal nula e é a 9ª mais importante do modelo.
11. **Métricas agregadas não bastam quando o uso é individual.** RF e XGBoost têm matriz de confusão
    idêntica e discordam sobre 24 alunos. Quando o produto é uma lista de nomes, são esses 24 que
    importam.
12. **Afirmações sobre blocos são robustas; sobre posições específicas, não.** Os três métodos de
    importância concordam com ρ de 0,70–0,77 — alto o suficiente para o bloco, baixo para a posição.

### 12.5 A contribuição, em uma frase

> O trabalho entrega menos do que "um sistema que prevê a evasão" e mais do que "um modelo com F1 de
> 0,91": entrega um **instrumento de priorização com explicação individual auditável**, acompanhado do
> mapa explícito de onde ele pode ser confiado e onde não pode.

Para uma coordenação de curso, a diferença entre receber uma lista de nomes e receber uma lista de nomes
**com o motivo de cada um** é a diferença entre saber que existe um problema e saber o que fazer a
respeito.

---

## 13. O que o trabalho não sustenta

### 13.1 Nada sobre causalidade

O SHAP mede **como o modelo usa cada variável**, não o efeito causal. Que a inadimplência contribua com
+1,86 em log-odds **não significa** que quitar a dívida reduza o risco em qualquer magnitude estimável.

Três recomendações são especialmente sensíveis:

| Recomendação | Problema causal |
|---|---|
| R-3 (bolsas) | **Viés de seleção** — bolsas exigem desempenho mínimo; bolsistas já são grupo selecionado |
| R-2 (inadimplência) | Pode ser **sintoma** da decisão de sair, não causa |
| R-4 (idade) | Idade é **marcador**, não causa — o que opera são as responsabilidades concorrentes |

### 13.2 Nada sobre alunos ativos

**A limitação mais séria.** O modelo foi treinado apenas em **trajetórias concluídas** (`Enrolled`
removido), mas seria aplicado a **alunos ativos** — precisamente a população excluída. O F1 é estimativa
válida para trajetórias concluídas, sem garantia de que se mantenha na população de aplicação.

### 13.3 Nada sobre aplicação precoce

As variáveis de desempenho do 1º e 2º semestres concentram a maior parte do poder preditivo e **só
existem ao final do primeiro ano**. A mediana de aprovações no 2º semestre entre evadidos é **zero** —
quando o modelo sinaliza, o aluno já parou de estudar.

> Parte do alto desempenho vem justamente de observar tarde. **Desempenho e utilidade não coincidem
> aqui.**

### 13.4 Nada sobre generalização geográfica

Base de uma única instituição portuguesa. Escala de notas 0–20, modalidades de ingresso portuguesas,
estrutura de mensalidades e classificação de ocupações distintas das brasileiras. **Os coeficientes e
limiares específicos não são transferíveis.** O que provavelmente se transfere são os **padrões
qualitativos**.

### 13.5 Nada sobre equidade comprovada

A auditoria **não encontrou** disparidade significativa — o que é diferente de comprovar equidade. Com
26 evadidos entre bolsistas, disparidades de 10 a 20 pp seriam indetectáveis.

### 13.6 Nada sobre custos assimétricos explícitos

O F1, adotado como critério, **pesa os dois erros igualmente** — o que o próprio trabalho nega. A métrica
teoricamente correta seria F-beta com β > 1 ou perda com custos monetários, que exigiria da instituição
uma estimativa do custo real de perder um aluno. A análise de limiar é o substituto pragmático.

### 13.7 Nada sobre validação temporal

O dataset cobre 2008 a 2019, mas a divisão treino/teste foi **aleatória**, não temporal. Uma divisão
temporal mediria a capacidade de extrapolação para o futuro, que é o uso real.

---

## 14. Trabalhos futuros

Nove propostas registradas em [`docs/07`](docs/07-limitacoes-e-trabalhos-futuros.md). Após as três
verificações finais, **quatro limitações foram resolvidas ou qualificadas** e as prioridades mudaram:

### Prioridade 1 — Modelagem de sobrevivência com dados censurados

**Resolve a limitação central** (§13.2). Tratar `Enrolled` como observação censurada à direita, em vez de
descartá-la — regressão de Cox, Kaplan-Meier, *random survival forests*.

Ganhos: usa os 794 alunos hoje descartados; responde **quando** o aluno tende a evadir, não apenas
**se**; elimina a diferença entre população de treino e de aplicação.

### Prioridade 2 — Modelo de alerta precoce

**Resolve §13.3.** Treinar um segundo modelo com **apenas** variáveis de matrícula. Desempenho
necessariamente inferior, mas aplicável **no primeiro dia de aula**. A comparação "quanto se ganha em
precisão por cada semestre de espera, e quantos alunos se perde nesse intervalo" seria um resultado
original. Esforço baixo — os dados já estão prontos.

### Prioridade 3 — Ampliar a auditoria de equidade

**Resolve §13.5.** Repetir a auditoria com volume acumulado maior e nos atributos ainda não examinados
(nacionalidade, escolaridade dos pais, curso). O notebook é reexecutável.

### Demais propostas

| Proposta | Resolve |
|---|---|
| Validação em dados brasileiros (a própria UEMG) | §13.4 |
| Otimização com custos assimétricos explícitos | §13.6 |
| Validação temporal por coorte de ingresso | §13.7 |
| Modelagem causal explícita (mediação) | §13.1 e a nuance da origem familiar |
| Comparação SMOTE × SMOTENC × `class_weight` | Ressalva sobre interpolação em categóricas |
| Protótipo de sistema de alerta com interface | Transforma a ferramenta da §8 em produto |

### Já endereçadas pelas verificações finais

| Limitação original | Situação |
|---|---|
| Partição única sem intervalo de confiança | **Resolvida** — *bootstrap* |
| Ausência de teste de significância | **Resolvida** — McNemar em dois escopos |
| Dependência de uma única semente | **Resolvida** — 10 sementes, viés quantificado |
| Validação cruzada não aninhada | **Qualificada** — efeito de segunda ordem |
| Ausência de auditoria de equidade | **Parcialmente resolvida** — 5 atributos auditados |
| Calibração de probabilidade (não prevista) | **Resolvida** — verificada e documentada |

---

## 15. Erros cometidos e corrigidos

Esta seção existe porque vários resultados do trabalho vieram de verificações que **encontraram erros** —
e o processo de encontrá-los é parte do método.

### Erros de dados

| Erro | Como foi detectado | Correção |
|---|---|---|
| 3.230 notas corrompidas no MySQL | Confronto entre tipo declarado (`TEXT`) e domínio esperado (0–20) | `reparar_notas()`, auditada contra o UCI |
| Regra de correção incompleta (deixaria 39 valores errados) | **Validação de faixa após a correção** | Regra reformulada para casos de ponto único |
| BOM corrompendo o nome da 1ª coluna | Inspeção do resultado (`imarital_status`) | `_slug()` remove BOM em ambas as formas |
| `nacionality` fora dos grupos conceituais (103/105 features cobertas) | Contagem de cobertura na própria célula | Grupo demográfico corrigido — 105/105 |
| Sufixo real do sklearn é `infrequent_sklearn`, não `infrequent_sublevel` | Rótulo aparecendo sem tradução nos gráficos | `rotulo()` corrigido |
| `.env.example` referenciado mas nunca criado | Conferência de entregáveis do Notebook 5 | Criado |

### Erros de análise

| Erro | Como foi detectado | Correção |
|---|---|---|
| Tolerância de `1e-9` para verificar a recomposição SHAP | A asserção falhou | Valores SHAP são `float32` (eps 1,2e-7); tolerância ajustada para `1e-4` |
| Gatilho de idade reportado como "≤ 70" em vez de "≥ 23" | Leitura da saída | A lógica assumia que toda variável é protetora; corrigida pelo sinal de ρ |
| "As 3 faixas mal calibradas são todas irrelevantes" | Conferência dos `n` por faixa | **Falso** para [0,1–0,2): 81 alunos, miscalibração material |
| "A ordenação dos modelos se mantém entre as sementes" | A saída mostrava LR passando de 2º para 3º | Substituído por contagem de lideranças por semente |
| "As correlações entre os três rankings são altas" | Valores reais: ρ de 0,70 a 0,77 | Reescrito como "moderadas a altas", com a distinção bloco × posição |
| Números de validação cruzada estimados em vez de conferidos | Extração dos artefatos | 5 valores corrigidos contra `metricas_modelos.csv` |
| Notação de valor absoluto em cabeçalho de tabela quebrava o markdown | Verificador de tabelas | Notação reescrita sem barras verticais |
| `sed` com alternância de BRE destruiu dois documentos | Revisão do resultado | A barra vertical escapada é alternância em BRE, não literal; arquivos reescritos |

> **A lição transversal:** em quase todos os casos, o que encontrou o erro foi uma **verificação
> automática que aborta a execução** — não a inspeção visual. Foi por isso que elas foram adicionadas em
> cada fase, e é por isso que valem mais do que uma afirmação em markdown: uma afirmação envelhece quando
> o código muda; uma asserção não.

---

## 16. Como reproduzir

```bash
# 1. Ambiente
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

# 2. Credenciais do banco (OPCIONAL)
#    Sem .env, o pipeline baixa os dados do UCI automaticamente.
copy .env.example .env
#    preencher MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE, MYSQL_TABLE

# 3. Executar os notebooks na ordem
jupyter lab
#    01 -> 02 -> 03 -> 04 -> 05    (pipeline principal)
#    06 -> 07 -> 08                (verificações finais, independentes entre si)
```

**Tempo total:** ~6 minutos. O gargalo é a busca de hiperparâmetros do Notebook 3 (1.760 ajustes),
paralelizada em `n_jobs=-1`.

**Reprodutibilidade:** `random_state=42` em toda operação estocástica. Executar na ordem produz
resultados idênticos aos documentados — o Notebook 7 verifica isso explicitamente, reconstruindo a
partição e conferindo contra os valores publicados.

**Segurança:** as credenciais nunca aparecem no código; são lidas de variáveis de ambiente via
`python-dotenv`, e o `.env` está no `.gitignore`.

---

## Documentação detalhada

| Documento | Conteúdo |
|---|---|
| [00 — Visão geral](docs/00-visao-geral.md) | Objetivo, dataset, arquitetura, ambiente |
| [01 — Entendimento dos dados](docs/01-entendimento-dos-dados.md) | Análise exploratória completa |
| [02 — Qualidade e integridade](docs/02-qualidade-e-integridade.md) | A falha do MySQL e sua correção auditada |
| [03 — Preparação dos dados](docs/03-preparacao-dos-dados.md) | `Enrolled`, codificação, split, SMOTE |
| [04 — Modelagem](docs/04-modelagem.md) | Três modelos, busca, comparação, limiar |
| [05 — Decisões metodológicas](docs/05-decisoes-metodologicas.md) | **40 decisões** com alternativa rejeitada e risco |
| [06 — Resultados e figuras](docs/06-resultados-e-figuras.md) | 27 tabelas e o índice das 28 figuras |
| [07 — Limitações e trabalhos futuros](docs/07-limitacoes-e-trabalhos-futuros.md) | 15 limitações, 9 propostas |
| [08 — Interpretabilidade com SHAP](docs/08-interpretabilidade-shap.md) | Importância, direção, casos individuais |
| [09 — Calibração de probabilidade](docs/09-calibracao.md) | Verificação 1 |
| [10 — Significância estatística](docs/10-significancia.md) | Verificação 2 |
| [11 — Equidade algorítmica](docs/11-equidade.md) | Verificação 3 |
| [12 — Revisão final](docs/12-revisao-final.md) | Changelog das correções de coerência interna |
| [13 — Aplicação a alunos ativos](docs/13-aplicacao-alunos-ativos.md) | O modelo sobre os 794 alunos que ainda cursam |

## Referências

- Realinho, V.; Machado, J.; Baptista, L.; Martins, M. V. (2022). *Predicting Student Dropout and
  Academic Success*. Data, 7(11), 146.
- Chawla, N. V.; Bowyer, K. W.; Hall, L. O.; Kegelmeyer, W. P. (2002). *SMOTE: Synthetic Minority
  Over-sampling Technique*. Journal of Artificial Intelligence Research, 16, 321–357.
- Chen, T.; Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. KDD '16.
- Lundberg, S. M.; Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS.
- Tinto, V. (1993). *Leaving College: Rethinking the Causes and Cures of Student Attrition*. 2ª ed.
- Cabrera, A. F.; Castañeda, M. B.; Nora, A.; Hengstler, D. (1992). *The Convergence Between Two Theories
  of College Persistence*. Journal of Higher Education, 63(2).
- Wirth, R.; Hipp, J. (2000). *CRISP-DM: Towards a Standard Process Model for Data Mining*.
