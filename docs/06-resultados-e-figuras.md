# 06 — Resultados e Figuras

> Tabelas e figuras consolidadas, formatadas para transcrição direta na monografia.
> Todos os números foram extraídos dos notebooks executados — nenhum valor é estimado.

> **Convenção de partição — vale para todas as tabelas deste documento.** Salvo indicação em
> contrário, todo número de desempenho vem da **partição da semente 42**. A
> [Verificação 2](10-significancia.md) mostrou que essa partição é **favorável**: em partições típicas
> o F1 é ≈ 0,88 (não 0,9059), o modelo detecta **84 de cada 100** evadidos (não 90, faixa 82–86) e
> perde 44 em vez de 29. A precisão não apresenta esse viés. **Comparações entre modelos permanecem
> válidas**, porque todos são avaliados sobre exatamente os mesmos alunos.

---

## 1. Caracterização da amostra

**Tabela 1 — Composição da amostra por situação acadêmica**

| Situação | n | % |
|---|---:|---:|
| Graduate — concluiu | 2.209 | 49,9 |
| Dropout — evadiu | 1.421 | 32,1 |
| Enrolled — matriculado | 794 | 17,9 |
| **Total** | **4.424** | **100,0** |

*Razão de desbalanceamento: 2,78:1. Fonte: base MySQL `tcc.data`, auditada contra UCI id=697.*

**Tabela 2 — Amostra após a formulação binária**

| Situação | n | % |
|---|---:|---:|
| Graduate | 2.209 | 60,9 |
| Dropout | 1.421 | 39,1 |
| **Total** | **3.630** | **100,0** |

---

## 2. Fatores associados à evasão

**Tabela 3 — Correlação ponto-bisserial com a evasão, 10 maiores em módulo**

| # | Variável | r | Direção do risco |
|---:|---|---:|---|
| 1 | Média das notas (2º sem.) | −0,572 | nota baixa → risco alto |
| 2 | Disciplinas aprovadas (2º sem.) | −0,570 | menos aprovações → risco alto |
| 3 | Média das notas (1º sem.) | −0,481 | nota baixa → risco alto |
| 4 | Disciplinas aprovadas (1º sem.) | −0,479 | menos aprovações → risco alto |
| 5 | Mensalidades em dia | −0,429 | inadimplência → risco alto |
| 6 | Idade no ingresso | +0,254 | idade maior → risco alto |
| 7 | Bolsista | −0,245 | bolsa → risco menor |
| 8 | Inadimplente | +0,229 | dívida → risco alto |
| 9 | Gênero | +0,204 | código 1 → risco alto |
| 10 | Avaliações realizadas (2º sem.) | −0,155 | menos avaliações → risco alto |

**Tabela 4 — Taxa de evasão segundo variáveis binárias** — n = 4.424, taxa-base 32,1%

| Variável | Valor 0 | Valor 1 | Diferença (pp) |
|---|---:|---:|---:|
| Mensalidades em dia | 86,6% | 24,7% | **−61,8** |
| Inadimplente | 28,3% | 62,0% | **+33,7** |
| Bolsista | 38,7% | 12,2% | **−26,5** |
| Gênero | 25,1% | 45,1% | +19,9 |
| Turno diurno/noturno | 42,9% | 30,8% | −12,1 |
| Deslocado da cidade de origem | 37,6% | 27,6% | −10,1 |
| Estudante internacional | 32,2% | 29,1% | −3,1 |
| Necessidades educacionais especiais | 32,1% | 33,3% | +1,2 |

> **Números para citar no texto:** entre alunos com mensalidades em atraso, a evasão é de **86,6%**,
> contra 24,7% dos que estão em dia. Bolsistas evadem a **12,2%**, contra 38,7% dos não bolsistas —
> menos de um terço do risco.

**Tabela 5 — Medianas por situação acadêmica**

| Variável | Dropout | Enrolled | Graduate |
|---|---:|---:|---:|
| Disciplinas aprovadas (2º sem.) | **0,0** | 4,0 | 6,0 |
| Média das notas (2º sem.) | **0,0** | 12,0 | 13,0 |
| Disciplinas aprovadas (1º sem.) | 2,0 | 5,0 | 6,0 |
| Média das notas (1º sem.) | 10,9 | 12,0 | 13,0 |
| Idade no ingresso | 23,0 | 20,0 | 19,0 |
| Nota de admissão | 123,6 | 124,1 | 127,4 |

**Tabela 6 — Importância das variáveis, top 10, calculada só no treino**

| # | Variável | Informação mútua | Importância RF |
|---:|---|---:|---:|
| 1 | Disciplinas aprovadas (2º sem.) | 0,3130 | 0,1778 |
| 2 | Disciplinas aprovadas (1º sem.) | 0,2587 | 0,1210 |
| 3 | Média das notas (2º sem.) | 0,2444 | 0,1078 |
| 4 | Média das notas (1º sem.) | 0,2088 | 0,0621 |
| 5 | Mensalidades em dia | 0,1087 | 0,0486 |
| 6 | Idade no ingresso | 0,0688 | 0,0307 |
| 7 | Avaliações realizadas (2º sem.) | 0,0849 | 0,0293 |
| 8 | Bolsista | 0,0572 | 0,0268 |
| 9 | Avaliações realizadas (1º sem.) | 0,0790 | 0,0249 |
| 10 | Nota de admissão | 0,0383 | 0,0225 |

*Concentração: top 5 = 51,7%; top 10 = 65,2%; top 20 = 80,0% da importância total.*

---

## 3. Multicolinearidade e outliers

**Tabela 7 — Pares de variáveis com correlação absoluta acima de 0,80**

| Par | r |
|---|---:|
| Disciplinas creditadas, 1º × 2º sem. | +0,945 |
| Disciplinas matriculadas, 1º × 2º sem. | +0,943 |
| Disciplinas aprovadas, 1º × 2º sem. | +0,904 |
| Média das notas, 1º × 2º sem. | +0,837 |

**Tabela 8 — Outliers pelo método IQR e sua associação com a evasão**

| Variável | Outliers | % da amostra |
|---|---:|---:|
| Média das notas (2º sem.) | 877 | 19,8 |
| Média das notas (1º sem.) | 726 | 16,4 |
| Disciplinas creditadas (1º sem.) | 577 | 13,0 |
| Ordem de opção do curso | 541 | 12,2 |
| Disciplinas creditadas (2º sem.) | 530 | 12,0 |
| Idade no ingresso | 441 | 10,0 |
| Disciplinas matriculadas (1º sem.) | 424 | 9,6 |
| Disciplinas matriculadas (2º sem.) | 369 | 8,3 |
| Disciplinas sem avaliação (1º sem.) | 294 | 6,6 |
| Disciplinas sem avaliação (2º sem.) | 282 | 6,4 |
| **Registros com ao menos um outlier** | **2.422** | **54,7** |

| | Taxa de evasão |
|---|---:|
| Registros **com** outlier | 45,7% |
| Registros **sem** outlier | 15,6% |
| **Diferença** | **+30,1 pp** |

---

## 4. Preparação dos dados

**Tabela 9 — Transformação da dimensionalidade**

| Etapa | Registros | Colunas / Features |
|---|---:|---:|
| Base auditada | 4.424 | 37 |
| Após remover `Enrolled` | 3.630 | 37 |
| Treino — split 80% | 2.904 | 36 |
| Teste — split 20% | 726 | 36 |
| Após codificação e padronização | — | **105** |
| Treino após SMOTE | **3.534** | 105 |

*105 features = 19 contínuas padronizadas + 8 binárias + 78 dummies, de 200 categorias agrupadas por `min_frequency=0,01`.*

**Tabela 10 — Efeito do SMOTE, apenas no treino**

| Classe | Antes | Depois | Sintéticos criados |
|---|---:|---:|---:|
| Graduate | 1.767 | 1.767 | 0 |
| Dropout | 1.137 | 1.767 | **+630** |
| **Total** | **2.904** | **3.534** | **630** |
| Razão | 1,55:1 | 1,00:1 | — |

*Conjunto de teste preservado: 726 registros, 39,1% de evasão em proporção natural.*

---

## 5. Desempenho dos modelos

**Tabela 11 — Piso trivial: classificadores que ignoram os dados** — teste, n = 726

| Estratégia | Acurácia | Precisão | Recall | F1 |
|---|---:|---:|---:|---:|
| Sempre Graduate | 0,6088 | 0,0000 | 0,0000 | 0,0000 |
| Sempre Dropout | 0,3912 | 0,3912 | 1,0000 | 0,5624 |
| Aleatório 50/50 | 0,4931 | 0,3901 | 0,5246 | 0,4474 |

**Tabela 12 — Desempenho no conjunto de teste** — n = 726, 39,1% de evasão, sem dados sintéticos

| Modelo | Acurácia | Acur. balanc. | Precisão | Recall | F1 | ROC-AUC | PR-AUC | MCC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Regressão Logística | 0,9132 | 0,9168 | 0,8576 | **0,9331** | 0,8938 | 0,9739 | 0,9710 | 0,8227 |
| Random Forest | 0,9270 | 0,9218 | 0,9140 | 0,8979 | 0,9059 | 0,9726 | 0,9697 | 0,8463 |
| **XGBoost** ✅ | **0,9270** | **0,9218** | **0,9140** | 0,8979 | **0,9059** | **0,9747** | **0,9741** | **0,8463** |

*Precisão, recall, F1, PR-AUC e MCC referem-se à classe `Dropout`.*

**Tabela 13 — Validação cruzada no treino** — `StratifiedKFold(5)`, média ± desvio-padrão

| Modelo | Acurácia | Precisão | Recall | F1 | ROC-AUC | Avg. precision |
|---|---|---|---|---|---|---|
| Regressão Logística | 0,8957 ± 0,0080 | 0,8765 ± 0,0130 | **0,8540 ± 0,0103** | 0,8651 ± 0,0099 | 0,9445 ± 0,0067 | 0,9415 ± 0,0032 |
| Random Forest | 0,9019 ± 0,0132 | 0,9097 ± 0,0191 | 0,8320 ± 0,0176 | 0,8691 ± 0,0175 | 0,9451 ± 0,0085 | 0,9427 ± 0,0088 |
| **XGBoost** | **0,9063 ± 0,0120** | **0,9163 ± 0,0123** | 0,8373 ± 0,0253 | **0,8748 ± 0,0171** | **0,9507 ± 0,0092** | **0,9465 ± 0,0063** |

**Tabela 14 — Hiperparâmetros selecionados**

| Modelo | Grade | Ajustes | Hiperparâmetros escolhidos |
|---|---:|---:|---|
| Regressão Logística | — padrão | 5 | `max_iter=1000` |
| Random Forest | 108 combinações | 540 | `n_estimators=100`, `max_depth=15`, `min_samples_split=2`, `min_samples_leaf=2` |
| XGBoost | 243 combinações | 1.215 | `n_estimators=200`, `max_depth=3`, `learning_rate=0.1`, `subsample=1.0`, `colsample_bytree=0.8` |

*Critério de refit: F1 da classe `Dropout`. Total: 1.760 ajustes de modelo.*

**Tabela 15 — Verificação de sobreajuste**

| Modelo | F1 na CV | F1 no teste | Diferença |
|---|---:|---:|---:|
| Regressão Logística | 0,8651 | 0,8938 | +0,0287 |
| Random Forest | 0,8691 | 0,9059 | +0,0368 |
| XGBoost | 0,8748 | 0,9059 | +0,0310 |

**Tabela 16 — Matriz de confusão do modelo final, XGBoost**

| | Previsto Graduate | Previsto Dropout |
|---|---:|---:|
| **Real Graduate** | 418 | 24 |
| **Real Dropout** | 29 | 255 |

| Categoria | n | Leitura operacional |
|---|---:|---|
| Verdadeiros positivos | 255 | Evadidos corretamente sinalizados |
| Falsos negativos | **29** | Evadidos não detectados — o erro caro |
| Falsos positivos | 24 | Concluintes sinalizados por engano |
| Verdadeiros negativos | 418 | Concluintes corretamente liberados |

> De cada 100 alunos que evadem, o modelo identifica **90 nesta partição dos dados**. De cada 100
> sinalizados, **91** realmente evadiriam. Total sinalizado: **279 de 726**, ou 38,4% da turma.
>
> **Ressalva de partição** — ver Tabela 28: os valores típicos entre dez partições são **84** detectados
> de cada 100 evadidos (faixa 82–86) e **44** falsos negativos (faixa 40–50). A precisão não apresentou o
> mesmo viés.

**Tabela 17 — Discordância entre modelos** — teste, n = 726

| Par | Alunos com classificação divergente |
|---|---:|
| Reg. Logística × Random Forest | 54 — 7,4% |
| Reg. Logística × XGBoost | 52 — 7,2% |
| Random Forest × XGBoost | 24 — 3,3% |
| **Concordância dos três** | **661 — 91,0%** |

*Random Forest e XGBoost têm matriz de confusão idêntica apesar de discordarem sobre 24 alunos.*

**Tabela 18 — Trade-off entre recall e precisão na escolha do modelo**

| Partição da semente 42 | Regressão Logística | XGBoost |
|---|---:|---:|
| Evadidos detectados | **265** de 284 | 255 de 284 |
| Evadidos perdidos | **19** | 29 |
| Falsos alarmes | 44 | **24** |
| Alunos convocados | 309 — 42,6% | **279 — 38,4%** |
| F1 | 0,8938 | **0,9059** |

---

## 6. Análise do limiar de decisão

**Tabela 19 — Cenários operacionais** — XGBoost, n = 726, 284 evadidos reais

| Limiar | Precisão | Recall | F1 | Sinalizados | Detectados | Não detectados | Falsos alarmes |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0,10 | 0,6219 | 0,9789 | 0,7606 | 447 | 278 | 6 | 169 |
| 0,20 | 0,7541 | 0,9718 | 0,8492 | 366 | 276 | 8 | 90 |
| **0,30** | 0,8242 | 0,9577 | 0,8860 | 330 | 272 | **12** | 58 |
| 0,40 | 0,8697 | 0,9401 | 0,9036 | 307 | 267 | 17 | 40 |
| **0,50** ⬅ adotado | **0,9140** | **0,8979** | **0,9059** | **279** | **255** | **29** | **24** |
| 0,60 | 0,9509 | 0,8873 | 0,9180 | 265 | 252 | 32 | 13 |
| **0,64** ⬅ F1 máximo | 0,9615 | 0,8803 | **0,9191** | 260 | 250 | 34 | 10 |
| 0,70 | 0,9680 | 0,8521 | 0,9064 | 250 | 242 | 42 | 8 |
| 0,80 | 0,9875 | 0,8345 | 0,9046 | 240 | 237 | 47 | 3 |

**Leituras para o texto:**

- Baixar o limiar de 0,50 para **0,30** reduz os evadidos não detectados de 29 para **12**, ao custo de
  34 conversas extras. **17 alunos resgatados.**
- Baixar para **0,10** deixa apenas 6 evadidos escaparem, mas convoca 447 alunos — 62% da turma,
  bem acima do teto de 50% adotado neste trabalho, e portanto fora da faixa em que a lista ainda
  funciona como priorização.
- **Elevar** o limiar é mau negócio: de 0,50 para 0,80, economizam-se 21 conversas ao custo de 18
  alunos perdidos.

---

## 7. Interpretabilidade — SHAP

**Tabela 20 — Importância global por SHAP**, top 10 de 105 features

| # | Variável | Impacto médio (SHAP absoluto) | % do total | % acumulado |
|---:|---|---:|---:|---:|
| 1 | Disciplinas aprovadas (2º sem.) | 1,5425 | 23,38 | 23,38 |
| 2 | Disciplinas aprovadas (1º sem.) | 0,8258 | 12,52 | 35,90 |
| 3 | Mensalidades em dia | 0,4232 | 6,42 | 42,32 |
| 4 | Disciplinas matriculadas (1º sem.) | 0,3918 | 5,94 | 48,26 |
| 5 | Média das notas (2º sem.) | 0,2509 | 3,80 | 52,06 |
| 6 | Bolsista | 0,2419 | 3,67 | 55,73 |
| 7 | Nota de admissão | 0,2401 | 3,64 | 59,37 |
| 8 | Idade no ingresso | 0,2032 | 3,08 | 62,45 |
| 9 | Taxa de desemprego | 0,1799 | 2,73 | 65,18 |
| 10 | Inadimplente | 0,1758 | 2,67 | 67,85 |

*24 das 105 features têm impacto nulo. Erro de aditividade: 3,8 × 10⁻⁶.*

**Tabela 21 — Direção do efeito** — ρ de Spearman entre valor da variável e seu SHAP

| Variável | Impacto | ρ | Efeito de valor alto |
|---|---:|---:|---|
| Disciplinas aprovadas (2º sem.) | 1,5425 | −0,967 | reduz o risco |
| Disciplinas aprovadas (1º sem.) | 0,8258 | −0,956 | reduz |
| Mensalidades em dia | 0,4232 | −0,604 | reduz |
| Disciplinas matriculadas (1º sem.) | 0,3918 | **+0,718** | **aumenta** |
| Média das notas (2º sem.) | 0,2509 | −0,894 | reduz |
| Bolsista | 0,2419 | −0,739 | reduz |
| Nota de admissão | 0,2401 | +0,254 | não monotônico |
| Idade no ingresso | 0,2032 | +0,786 | aumenta |
| Taxa de desemprego | 0,1799 | **+0,943** | **aumenta** |
| Inadimplente | 0,1758 | +0,554 | aumenta |
| Avaliações realizadas (2º sem.) | 0,1692 | +0,639 | aumenta |
| Deslocado da cidade de origem | 0,1425 | **+0,861** | **aumenta** |

**Tabela 22 — Efeito marginal × efeito condicional**, o achado principal da Fase 5

| Variável | Marginal | Condicional | Situação |
|---|---|---|---|
| Deslocado da cidade de origem | r = −0,126, protege: 33,6% × 45,9% | ρ = +0,861 | inversão de sinal |
| Disciplinas matriculadas (1º sem.) | r = −0,161, protege | ρ = +0,718, 4ª mais importante | inversão de sinal |
| Taxa de desemprego | r = −0,004, nulo | ρ = +0,943, 9ª mais importante | efeito que emerge |

**Tabela 23 — Contribuição por grupo conceitual**

| # | Grupo | Features | Impacto médio | % do total |
|---:|---|---:|---:|---:|
| 1 | Desempenho acadêmico | 12 | 2,4089 | 59,9 |
| 2 | Situação financeira | 3 | 0,5992 | 14,9 |
| 3 | Trajetória de ingresso | 36 | 0,3809 | 9,5 |
| 4 | Perfil demográfico | 11 | 0,2292 | 5,7 |
| 5 | Contexto macroeconômico | 3 | 0,2066 | 5,1 |
| 6 | Origem familiar | 40 | 0,1960 | 4,9 |

*Cobertura: 105 de 105 features. Desempenho acadêmico entrega 59,9% do impacto com 12 features; origem familiar, 4,9% com 40.*

**Tabela 24 — Ganho da recomposição das variáveis nominais**

| Variável | Dummies | Melhor dummy | Recomposta | Ganho |
|---|---:|---:|---:|---:|
| Ocupação do pai | 13 | 0,0466 | 0,1174 | 2,5× |
| Curso | 17 | 0,0965 | 0,2332 | 2,4× |
| Escolaridade da mãe | 8 | 0,0303 | 0,0603 | 2,0× |
| Ocupação da mãe | 11 | 0,0492 | 0,0821 | 1,7× |
| Escolaridade do pai | 8 | 0,0363 | 0,0596 | 1,6× |
| Modalidade de ingresso | 10 | 0,0810 | 0,1128 | 1,4× |
| Formação anterior | 6 | 0,0868 | 0,0893 | 1,0× |
| Nacionalidade | 2 | 0,0012 | 0,0012 | 1,0× |
| Estado civil | 4 | 0,0000 | 0,0000 | impacto nulo |

**Tabela 25 — Gatilhos operacionais em unidades reais**

| Variável | Gatilho de alerta | SHAP no pior valor |
|---|---|---:|
| Disciplinas aprovadas (2º sem.) | até **3** disciplinas | +2,43 com 0 aprovações |
| Disciplinas aprovadas (1º sem.) | até **4** disciplinas | +1,04 |
| Média das notas (2º sem.) | até **11** na escala 0–20 | +0,36 |
| Idade no ingresso | **23** anos ou mais | +0,51 com 25 anos |

**Tabela 26 — Convergência entre os três métodos de importância**

| Par de métodos | ρ de Spearman |
|---|---:|
| Correlação (NB1) × impureza (NB2) | +0,753 |
| Impureza (NB2) × SHAP (NB4) | +0,774 |
| Correlação (NB1) × SHAP (NB4) | +0,697 |

**Tabela 27 — Os três casos individuais**

| Caso | Prob. evasão | Real | Predito | Fator dominante |
|---|---:|---|---|---|
| A — alto risco | 1,000 | Dropout | Dropout | Aprovações 2º sem., +2,291 |
| B — baixo risco | 0,004 | Graduate | Graduate | Aprovações 2º sem., −1,405 |
| **C — falso negativo** | 0,023 | **Dropout** | Graduate | Aprovações 2º sem., **−1,610** |

*O caso C evadiu com perfil dominado por fatores protetores — evidência do teto intrínseco imposto pelas variáveis disponíveis.*

---

**Tabela 28 — Números operacionais em duas versões** — XGBoost, matrizes de confusão reais

| Métrica | Semente 42 (reportada) | Média das 10 sementes | Faixa |
|---|---:|---:|---|
| F1 | 0,9059 | 0,8796 | 0,8656 – 0,8909 |
| Recall | 0,8979 | **0,8437** | 0,8239 – 0,8592 |
| Precisão | 0,9140 | 0,9191 | 0,8873 – 0,9412 |
| Detectados de cada 100 evadidos | **90** | **84** | 82 – 86 |
| Acerto de cada 100 sinalizados | **91** | **92** | 89 – 94 |
| Verdadeiros positivos | 255 | 240 | 234 – 244 |
| Falsos negativos | **29** | **44** | 40 – 50 |
| Falsos positivos | 24 | 21 | 15 – 31 |
| Alunos sinalizados | 279 | 261 | 253 – 275 |

*A coluna "média das 10" reporta a média das métricas de cada execução, não a métrica recalculada das contagens médias — as duas diferem por 0,001 a 0,002 porque F1, recall e precisão são razões. Ver [doc 10](10-significancia.md), §4.5.*

**Tabela 29 — Custo pareado dos dois modelos** — curva exata, sem interpolação

| Região | Recall | Pontos | Melhor modelo | Convocados |
|---|---|---:|---|---|
| Estável | 0,852 – 0,9577 | 31 | **XGBoost em 31 de 31** | 34% a 46% da turma |
| Banda estreita | 0,9613 – 0,9718 | 4 | **Reg. Logística em 4 de 4** | 47% a 49% da turma |
| Instável | ≥ 0,9754 | 8 | oscila — 3 travessias de liderança | 59% a 94% da turma |

*Convocação considerando os dois modelos. A curva tem 9 pontos vencidos pela Regressão Logística e
3 travessias de liderança — são contagens distintas. Abaixo do teto de 50%, o XGBoost vence 72 dos
76 pontos decididos.*

**Tabela 30 — Comparação isocusto** — mesmo número de falsos alarmes

| Falsos alarmes | XGBoost detecta | Reg. Logística detecta | Diferença |
|---:|---:|---:|---:|
| 24 | 255 | 254 | +1 |
| 35 | 266 | 261 | +5 |
| 40 | 267 | 265 | +2 |
| 44 | 267 | 266 | +1 |
| 58 | 272 | 269 | +3 |

**Tabela 31 — PR-AUC com intervalo de confiança** — 1.000 reamostragens

| Modelo | PR-AUC | IC 95% |
|---|---:|---|
| Regressão Logística | 0,9710 | [0,9587 — 0,9816] |
| Random Forest | 0,9697 | [0,9567 — 0,9811] |
| XGBoost | 0,9741 | [0,9623 — 0,9846] |

*Diferença pareada XGBoost − Reg. Logística: +0,0031, IC 95% [−0,0027, +0,0093] — contém o zero, não significativa.*

---

### Tabela 32 — Perfil dos falsos negativos contra os demais grupos

| Variável | Falsos negativos | Evadidos detectados | Formandos | Parece com |
|---|---:|---:|---:|---|
| Disciplinas aprovadas (2º sem.) | **6,24** | 1,23 | 6,10 | formando |
| Disciplinas aprovadas (1º sem.) | **6,76** | 1,82 | 6,12 | formando |
| Média das notas (2º sem.) | **12,37** | 5,00 | 12,55 | formando |
| Mensalidades em dia | **0,97** | 0,62 | 0,99 | formando |
| Inadimplente | **0,07** | 0,24 | 0,05 | formando |
| Idade no ingresso | 24,55 | 26,39 | 21,60 | evadido |
| Bolsista | 0,07 | 0,09 | 0,33 | evadido |

*Em 5 das 7 variáveis-chave o falso negativo médio está mais próximo do formando do que do evadido que
o modelo detectou. Fonte: `dados/perfil_falsos_negativos.csv`.*

### Tabela 33 — Taxa condicional de falso negativo: o núcleo duro

| Taxa condicional | Alunos | % |
|---|---:|---:|
| Nunca errado (0%) | 398 | 80,7% |
| Baixa (0 < t ≤ 30%) | 4 | 0,8% |
| Intermediária (30–99%) | 23 | 4,7% |
| **Sempre errado (100%)** | **68** | **13,8%** |

*Base: 493 evadidos que apareceram no conjunto de teste ao menos 3 vezes ao longo das 10 partições. A
taxa é condicional às aparições, e não à contagem simples entre sementes — esta última é confundida
pela composição do split. Erro aleatório à taxa global de 15,6% produziria 1,3 alunos com taxa de
100%; observam-se 68, cerca de 54 vezes mais. Fonte:
`dados/taxa_condicional_falsos_negativos.csv`.*

---

## 8. Índice de figuras

Todas em PNG, 200 dpi, prontas para inserção.

| Arquivo | Sugestão de legenda | Onde usar |
|---|---|---|
| `figuras/01_distribuicao_target.png` | Distribuição da variável-alvo, n = 4.424: barras absolutas e participação relativa | Caracterização da amostra |
| `figuras/02_histogramas_continuas.png` | Distribuição das variáveis contínuas — note a bimodalidade das notas semestrais | Análise exploratória |
| `figuras/03_matriz_correlacao.png` | Matriz de correlação de Pearson entre as variáveis contínuas | Análise exploratória |
| `figuras/04_top10_correlacao_evasao.png` | Dez variáveis mais associadas à evasão, correlação ponto-bisserial | Fatores de risco |
| `figuras/05_boxplots_por_classe.png` | Distribuição das principais variáveis por situação acadêmica | Fatores de risco |
| `figuras/06_efeito_smote.png` | Efeito do SMOTE aplicado exclusivamente ao treino; teste preservado em proporção natural | Metodologia — balanceamento |
| `figuras/07_importancia_features.png` | Importância das variáveis: informação mútua e redução de impureza | Metodologia — diagnóstico |
| `figuras/08_matrizes_confusao.png` | Matrizes de confusão dos três modelos no conjunto de teste | Resultados |
| `figuras/09_curvas_roc_pr.png` | Curvas ROC e Precisão-Revocação comparativas | Resultados |
| `figuras/10_comparacao_metricas.png` | Comparação de desempenho dos três modelos, com o piso trivial de acurácia | Resultados |
| `figuras/11_analise_limiar.png` | Ajuste do limiar de decisão: métricas e custo operacional em número de alunos | Discussão — aplicação prática |
| `figuras/12_shap_importancia_global.png` | Importância global das variáveis pelo valor SHAP médio absoluto | Interpretabilidade |
| `figuras/13_shap_beeswarm.png` | Impacto e direção de cada variável; cada ponto é um aluno | Interpretabilidade |
| `figuras/14_shap_impacto_direcao.png` | Impacto médio com a direção do efeito codificada em cor | Interpretabilidade |
| `figuras/15_shap_variaveis_recompostas.png` | Importância por variável original, com as dummies somadas | Interpretabilidade |
| `figuras/16_shap_convergencia_metodos.png` | Concordância entre importância por impureza e valor SHAP | Discussão metodológica |
| `figuras/17_shap_dependencia_top3.png` | Gráficos de dependência das três variáveis mais importantes | Fatores de risco |
| `figuras/18_shap_waterfall_a.png` | Decomposição da predição de um aluno de alto risco | Aplicação individual |
| `figuras/18_shap_waterfall_b.png` | Decomposição da predição de um aluno de baixo risco | Aplicação individual |
| `figuras/18_shap_waterfall_c.png` | Decomposição de um **falso negativo** — onde o modelo falha | Limitações |
| `figuras/19_shap_force_a.png` a `_c.png` | Os mesmos três casos em formato compacto, *force plot* | Aplicação individual |
| `figuras/20_shap_grupos_conceituais.png` | Participação de cada grupo conceitual no impacto total | Resultados |
| `figuras/21_painel_consolidado.png` | Painel de resultados consolidados do modelo final | Resultados |
| `figuras/22_curva_calibracao.png` | Curva de calibração com IC por faixa e efeito da calibração posterior | Verificação — calibração |
| `figuras/23_intervalos_confianca.png` | *Bootstrap*, ICs por métrica e variabilidade entre sementes | Verificação — significância |
| `figuras/24_equidade_subgrupos.png` | Recall e taxa de falsos positivos por subgrupo | Verificação — equidade |
| `figuras/25_curvas_custo_pareadas.png` | Custo de cada modelo em pontos de operação pareados, com teto de convocação | Discussão — escolha de modelo |
| `figuras/26_alunos_ativos_distribuicao.png` | Onde os alunos ativos caem na escala do modelo, contra evadidos e formandos | Aplicação — alunos ativos |
| `figuras/27_aluno_ativo_waterfall.png` | Decomposição do aluno ativo de maior risco | Aplicação — alunos ativos |

> **Recomendação para a apresentação oral.** Três figuras carregam o argumento:
>
> 1. `11_analise_limiar.png` — o painel da direita converte métricas abstratas em contagens de alunos,
>    o que torna o trade-off imediato para uma banca não especializada em ML.
> 2. `13_shap_beeswarm.png` — mostra impacto e direção de tudo em um único gráfico; é a figura que
>    melhor resume o resultado do trabalho.
> 3. `18_shap_waterfall_c.png` — o falso negativo. Apresentar a própria falha do modelo, explicada, é
>    mais convincente do que apresentar só acertos.

---

## 9. Números-chave para o resumo e a conclusão

| Afirmação | Número |
|---|---|
| Amostra total | 4.424 estudantes |
| Taxa de evasão na amostra | 32,1% — 39,1% na formulação binária |
| Valores de nota corrompidos no banco e reconstruídos | 3.230 |
| Auditoria contra a fonte oficial | 37/37 colunas idênticas, 0 divergências |
| Features após codificação | 105 |
| Ajustes de modelo na busca de hiperparâmetros | 1.760 |
| **F1-score do modelo final, classe Dropout** | **0,9059** |
| **Recall do modelo final** | **0,8979** |
| ROC-AUC do modelo final | 0,9747 |
| Ganho do XGBoost sobre a linha de base linear | **+1,4% em F1** |
| Evadidos identificados de cada 100 | 90 nesta partição · **84 típico** (82–86) |
| Acerto entre os sinalizados | 91 de cada 100 · 92 típico (89–94) |
| Concordância entre os três modelos | 91,0% dos alunos |
| Diferença de evasão: mensalidades atrasadas × em dia | 86,6% × 24,7% — **+61,8 pp** |
| Diferença de evasão: não bolsistas × bolsistas | 38,7% × 12,2% — **−26,5 pp** |
| Diferença de evasão: com outlier × sem outlier | 45,7% × 15,6% — **+30,1 pp** |
| Mediana de disciplinas aprovadas no 2º sem., Dropout × Graduate | 0 × 6 |
| Contribuições SHAP calculadas | 76.230 — 726 alunos × 105 features |
| Erro de aditividade do SHAP | 3,8 × 10⁻⁶ |
| Participação do desempenho acadêmico no impacto total | **59,9%** com 12 features |
| Participação da origem familiar | 4,9% com 40 features |
| Variável mais importante, isolada | 23,4% do impacto total |
| Features com impacto nulo | 24 de 105 |
| Gatilho operacional principal | **até 3 aprovações no 2º semestre** |
| Variáveis com efeito condicional oposto ao marginal | **3** |
