# 09 — Calibração de Probabilidade

> **Notebook:** `06_calibracao.ipynb` (15 células)
> **Motivação:** verificar se a análise de limiar de decisão do [doc 04](04-modelagem.md), §10, se
> sustenta — ela trata a saída do XGBoost como probabilidade real de evasão.
> **Saída:** `figuras/22_curva_calibracao.png`
> **Resultado:** o modelo original é mantido; a linguagem probabilística recebe ressalva.

---

## 1. Por que a verificação era necessária

A análise de limiar afirma coisas como *"baixar o limiar para 0,30 resgata 17 alunos"*, e o
[doc 08](08-interpretabilidade-shap.md) apresenta casos individuais com *"probabilidade de evasão de
0,023"*. Ambas as formulações pressupõem que, entre os alunos a quem o modelo atribui 0,30, cerca de
30% realmente evadam — isto é, que o modelo esteja **calibrado**.

**Essa premissa não é automática**, e havia um indício concreto de que poderia falhar: o valor base do
SHAP ficou em **0,5025** (doc 08, §2), enquanto a taxa real de evasão no teste é de **39,1%**. Um
modelo treinado sobre dados balanceados por SMOTE aprende a distribuição artificial 50/50, não a real.

A lacuna não estava registrada em [07 — Limitações](07-limitacoes-e-trabalhos-futuros.md): foi
identificada depois, e este documento a fecha.

### O que dependia da resposta

| Afirmação do trabalho | Depende de calibração? |
|---|---|
| "F1 de 0,9059 na classe `Dropout`" | **Não** — depende da ordenação e do limiar, não do valor numérico |
| "Baixar o limiar de 0,50 para 0,30 resgata 17 alunos" | **Era o que se precisava verificar** |
| "Este aluno tem 85% de probabilidade de evadir" | **Sim** — é afirmação probabilística direta |

---

## 2. Diagnóstico agregado

| Medida | Valor |
|---|---|
| Brier score | **0,0520** |
| ECE (10 faixas) | **0,1052** |
| Probabilidade média predita | 0,4284 |
| Taxa real de evasão no teste | 0,3912 |
| IC 95% da taxa real (Wilson) | [0,3563 — 0,4272] |
| **Diferença** | **+3,7 pontos percentuais** |

**A probabilidade média predita cai fora do intervalo de confiança da taxa real** — por pouco (0,4284
contra o limite superior de 0,4272), mas fora. O modelo **superestima sistematicamente** o risco médio.

A causa é conhecida e documentada: é o efeito esperado do SMOTE ([D-15](05-decisoes-metodologicas.md)).
Não é um defeito descoberto agora — é uma consequência quantificada de uma decisão registrada.

---

## 3. Onde está o erro — o diagnóstico por faixa

O ECE agregado esconde a informação decisiva: **em que faixa o erro ocorre, e quantos alunos há nessa
faixa.** Um desvio de 0,34 com 14 alunos e um desvio de 0,02 com 279 alunos não são comparáveis.

Cada faixa recebeu o **intervalo de confiança de Wilson** da frequência observada. Se o IC contém a
probabilidade predita, o desvio é compatível com ruído amostral.

| Faixa | n | % do teste | Predito | Observado | IC 95% | IC contém? |
|---|---:|---:|---:|---:|---|---|
| [0,0 – 0,1) | **279** | 38,4 | 0,043 | 0,022 | [0,010 – 0,046] | sim |
| [0,1 – 0,2) | **81** | 11,2 | 0,142 | **0,025** | [0,007 – 0,086] | **NÃO** |
| [0,2 – 0,3) | 36 | 5,0 | 0,244 | 0,111 | [0,044 – 0,253] | sim |
| [0,3 – 0,4) | 23 | 3,2 | 0,350 | 0,217 | [0,097 – 0,419] | sim |
| [0,4 – 0,5) | 28 | 3,9 | 0,454 | 0,429 | [0,265 – 0,609] | sim |
| [0,5 – 0,6) | 14 | 1,9 | 0,554 | 0,214 | [0,076 – 0,476] | **NÃO** |
| [0,6 – 0,7) | 15 | 2,1 | 0,656 | 0,667 | [0,417 – 0,848] | sim |
| [0,7 – 0,8) | 10 | 1,4 | 0,753 | 0,500 | [0,237 – 0,763] | sim |
| [0,8 – 0,9) | 21 | 2,9 | 0,856 | 0,857 | [0,654 – 0,950] | sim |
| [0,9 – 1,0) | **219** | 30,2 | 0,981 | 1,000 | [0,983 – 1,000] | **NÃO** |

### 3.1 A distribuição é bimodal

**Dois terços dos alunos (498 de 726, ou 68,6%) caem nas duas faixas extremas** — e ali a calibração é
praticamente perfeita (0,043 × 0,022 e 0,981 × 1,000).

O modelo não distribui probabilidades uniformemente: **ele é decidido**. Isso é coerente com o que a
Fase 5 mostrou — o modelo aprendeu fronteiras abruptas, em degrau, típicas de árvores. Um modelo que
separa bem produz probabilidades polarizadas.

### 3.2 As três faixas mal calibradas não são equivalentes

Esta distinção é o ponto central do documento:

| Faixa | n | Predito × observado | Natureza do problema |
|---|---:|---|---|
| [0,9 – 1,0) | 219 | 0,981 × 1,000 | Desvio de 2 pp — **irrelevante na prática** |
| [0,5 – 0,6) | 14 | 0,554 × 0,214 | **Ruído amostral** — com 14 alunos o IC é largo demais para concluir |
| **[0,1 – 0,2)** | **81** | **0,142 × 0,025** | **Miscalibração real e material** |

A faixa **[0,1 – 0,2)** é o único caso preocupante: são **81 alunos (11,2% do teste)** a quem o modelo
atribui cerca de 14% de risco quando a frequência real é de **2,5%**. O modelo é **excessivamente
cauteloso** nessa região.

> **Consequência operacional concreta.** Essa é uma razão adicional para não baixar o limiar
> agressivamente. Um corte em 0,10 convocaria esses 81 alunos, dos quais apenas 2 evadiriam — o que
> explica, em termos de calibração, por que a tabela de cenários do [doc 04](04-modelagem.md) mostra
> **169 falsos alarmes** naquele ponto.
>
> Para um sistema de triagem, errar por excesso de cautela é o lado seguro do erro. Mas o custo em
> convocações desnecessárias é real, e agora está quantificado.

---

## 4. Calibração posterior

Dois métodos padrão, ambos ajustados **exclusivamente sobre o treino sem SMOTE**
(`X_train_sem_smote.csv`), com `CalibratedClassifierCV(cv=5)`.

### Por que sobre o treino natural, e não sobre o teste

Calibrar no teste e depois reportar métricas nele seria **vazamento** — o calibrador teria visto os
rótulos que a métrica finge não conhecer. E calibrar no treino **com** SMOTE seria pior: ensinaria o
calibrador a reproduzir a proporção artificial 50/50, que é a origem do problema.

Com `cv=5` sobre o treino natural, em cada dobra o pipeline (SMOTE incluído) treina em 4/5 e o
calibrador é ajustado no 1/5 restante, que nunca foi reamostrado.

### Resultado

| Método | Brier | ECE | Prob. média | ROC-AUC | F1 @0,50 | Recall @0,50 | Precisão @0,50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **sem calibração** | 0,0520 | 0,1052 | 0,4284 | 0,9747 | **0,9059** | 0,8979 | 0,9140 |
| sigmoid (Platt) | 0,0523 | 0,1189 | 0,4242 | 0,9747 | 0,9046 | **0,9014** | 0,9078 |
| isotonic | **0,0516** | **0,0880** | 0,4262 | 0,9747 | 0,9071 | 0,8944 | **0,9203** |

**O ROC-AUC é idêntico nos três casos.** Isso não é coincidência: a calibração é uma transformação
**monotônica** das probabilidades, portanto **não altera a ordenação dos alunos** — apenas o valor
numérico atribuído a cada um.

### Decisão: não adotar

O modelo original em `modelos/melhor_modelo.joblib` permanece **inalterado**, e nenhum artefato novo de
modelo foi gravado. Três razões:

1. **O ganho é desprezível.** O Brier melhora **0,0004** com isotonic — e o Brier é a métrica
   apropriada, porque é uma *regra de pontuação própria*: penaliza calibração e discriminação
   conjuntamente. O sigmoid, inclusive, **piora** tanto o Brier quanto o ECE.
2. **A calibração não resolve o problema real.** A miscalibração se concentra na faixa central, pouco
   populada; nem o isotonic nem o sigmoid têm dados suficientes ali para corrigir de forma confiável.
   Ambos deixam o erro praticamente onde estava.
3. **Introduziria uma camada a mais sem benefício mensurável**, contra a orientação de manter o
   pipeline auditável.

Reproduzir a versão calibrada exige apenas as duas linhas da seção 4 do notebook, caso se queira
reportá-la como análise de sensibilidade.

---

## 5. Veredito sobre a análise de limiar

### O que NÃO muda

**A análise de limiar do [doc 04](04-modelagem.md), §10, permanece válida exatamente como está.**

A razão é que aquela análise **não é uma inferência probabilística**. Ela varre o limiar e, para cada
valor, **conta alunos reais**: quantos foram sinalizados, quantos evadiram, quantos escaparam. Essas
contagens são medidas empíricas diretas no conjunto de teste. Elas dependem apenas da **ordenação** dos
alunos — que a calibração não altera.

A verificação empírica confirma a afirmação original com precisão:

| | |
|---|---|
| Alunos com escore entre 0,30 e 0,50 | **51** |
| Dos quais realmente evadiram | **17** |
| Dos quais não evadiram | **34** |

O [doc 04](04-modelagem.md) afirmava: *"baixar para 0,30 reduz os evadidos não detectados de 29 para 12
— 17 alunos resgatados ao custo de 34 conversas extras"*. **Confirmado exatamente**: 17 evadidos e 34
não evadidos naquela faixa.

O que a tabela chama de "limiar 0,30" é um **ponto de corte na ordenação**, não uma afirmação de que
aqueles alunos têm 30% de risco.

### O que MUDA

**A linguagem probabilística precisa de ressalva.** Afirmações do tipo *"este aluno tem 85% de
probabilidade de evadir"* não se sustentam com o rigor que sugerem.

| Formulação a evitar | Formulação correta |
|---|---|
| "probabilidade de evasão de 85%" | "**escore de risco** de 0,85" |
| "30% dos alunos nesta faixa evadem" | "esta faixa corresponde ao ponto de corte 0,30 na ordenação" |

Onde o texto usar "probabilidade", cabe registrar que o modelo **superestima levemente e de forma
sistemática** o risco médio (+3,7 pp), efeito esperado do treino balanceado por SMOTE.

---

## 6. Síntese

| Verificação | Resultado |
|---|---|
| Brier score | 0,0520 — boa qualidade probabilística agregada |
| ECE (10 faixas) | 0,1052 — moderado, concentrado na faixa central |
| Probabilidade média vs taxa real | superestima em 3,7 pp, **fora** do IC de 95% |
| Faixas com miscalibração detectável | 3 de 10 — duas irrelevantes (ruído ou desvio de 2 pp); uma **material**: [0,1 – 0,2), com 81 alunos |
| Distribuição das probabilidades | **bimodal** — 68,6% dos alunos nos extremos, onde a calibração é quase perfeita |
| Calibração posterior | ganho de Brier desprezível (0,0004); ROC-AUC inalterado |
| **Decisão** | **manter o modelo original**; adicionar ressalva à linguagem probabilística |

### Para a monografia

1. **A ordenação dos alunos é confiável; o valor numérico é aproximado.** Chamar a saída de "escore de
   risco" em vez de "probabilidade" é mais honesto e **não enfraquece nenhum resultado** — o F1, o
   recall e a análise de limiar todos dependem só da ordenação.
2. **A análise de limiar não foi afetada**, porque conta alunos reais em vez de inferir probabilidades.
   Vale explicitar esse argumento no texto: é exatamente o tipo de detalhe que uma banca sonda.
3. **O modelo superestima levemente o risco médio**, e a causa é conhecida — o SMOTE. Consequência
   documentada da decisão D-15, não defeito novo.
4. **A distribuição bimodal é um resultado por si só.** Mostra que o modelo é decidido na maioria dos
   casos, e que a zona de incerteza — onde a calibração do limiar realmente opera — envolve uma minoria
   dos alunos.
5. **Há uma miscalibração material na faixa de risco baixo-mas-não-nulo** ([0,1 – 0,2), 81 alunos): o
   modelo atribui ~14% onde a frequência real é 2,5%. O erro é para o lado seguro, mas quantifica o
   custo de baixar o limiar agressivamente e deve constar como ressalva à recomendação **R-7**.

### Decisão registrada

**D-38 · Manter o modelo não calibrado, com ressalva de linguagem**

| | |
|---|---|
| **Alternativa rejeitada** | Adotar `CalibratedClassifierCV` com isotonic e substituir o modelo final |
| **Justificativa** | Ganho de Brier de 0,0004; a miscalibração se concentra em faixa pouco populada onde nenhum calibrador tem dados suficientes; o ROC-AUC é idêntico, logo a ordenação — de que todos os resultados dependem — não muda |
| **Risco / limitação** | Afirmações probabilísticas diretas ficam imprecisas; resolvido chamando a saída de "escore de risco". Superestimação sistemática de 3,7 pp registrada |
| **Onde** | `06_calibracao.ipynb` §4–5 |
