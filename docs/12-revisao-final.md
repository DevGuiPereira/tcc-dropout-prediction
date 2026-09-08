# 12 — Revisão Final: Changelog das Correções

> **Convenção de partição — vale para todas as tabelas deste documento.** Salvo indicação em
> contrário, todo número de desempenho vem da **partição da semente 42**. A
> [Verificação 2](10-significancia.md) mostrou que essa partição é **favorável**: em partições típicas
> o F1 é ≈ 0,88 (não 0,9059), o modelo detecta **84 de cada 100** evadidos (não 90, faixa 82–86) e
> perde 44 em vez de 29. A precisão não apresenta esse viés. **Comparações entre modelos permanecem
> válidas**, porque todos são avaliados sobre exatamente os mesmos alunos.

> Registro das correções aplicadas após duas rodadas de revisão crítica das três verificações finais
> (calibração, significância, equidade).
>
> **Nenhuma conclusão qualitativa do trabalho mudou.** Todas as correções são de **precisão de comparação
> e honestidade de comunicação**. O modelo final permanece o XGBoost, os fatores de risco permanecem os
> mesmos, e a tese central — o sinal está nos dados, não no algoritmo — sai **reforçada**.

---

## Sumário das rodadas

| Rodada | Objeto | Correções |
|---|---|---:|
| **1ª** | Contradição do critério F1 × premissa de custo; propagação do viés de semente | 7 ajustes |
| **2ª** | Refinamento das evidências e da linguagem | 3 correções |

---

## Rodada 1 — sete ajustes

### Ajuste 1 · Comparação de custo desigual

**O que estava errado.** A tabela comparava *"ajustar o limiar: +17 evadidos"* contra *"trocar de modelo:
+10 evadidos"* — mas as duas intervenções pagam preços diferentes (58 e 44 falsos alarmes). Concluir que
uma "rende mais" sem equalizar o custo é comparação inválida.

**O que foi corrigido.** Substituída pela comparação **isocusto**, fixando o número de falsos alarmes:

| Falsos alarmes | XGBoost detecta | Reg. Logística detecta | Diferença |
|---:|---:|---:|---:|
| 24 | 255 | 254 | +1 |
| 35 | 266 | 261 | +5 |
| 40 | 267 | 265 | +2 |
| 44 | 267 | 266 | +1 |
| 58 | 272 | 269 | +3 |

Reportada em **cinco pontos**, não um só: a vantagem oscila entre +1 e +5 evadidos, e reportar um único
valor esconderia essa variação.

**Mudou alguma conclusão?** A magnitude da vantagem do XGBoost passou de "aparentemente relevante" para
**explicitamente marginal**. A conclusão qualitativa (os modelos são próximos) não mudou — ficou mais bem
sustentada.

**Onde:** `docs/04-modelagem.md` §9.4 · `RELATORIO-CONSOLIDADO.md` §6.6 · `03_treinamento_modelos.ipynb` §9

---

### Ajuste 2 · Comparação assimétrica com seleção pós-hoc no teste

**O que estava errado, em dois níveis.** (a) O limiar 0,43 do XGBoost foi **buscado no próprio conjunto de
teste** para produzir o ponto de comparação — seleção pós-hoc no mesmo dado usado para avaliar. (b) A
comparação era **assimétrica**: só o limiar do XGBoost foi ajustado, a Regressão Logística ficou parada
em 0,50.

**O que foi corrigido.**

1. Gerada a curva de limiar completa da Regressão Logística, nos mesmos limiares do XGBoost, para que os
   dois modelos pudessem ser comparados ponto a ponto. Essa grade foi depois **superada** pela curva de
   custo exata da Correção 3, que é a versão persistida em `dados/curva_custo_exata.csv`.
2. Comparação refeita em **pontos de recall pareados**, com os dois modelos otimizados.
3. Os números da versão antiga (limiar 0,43, "35 contra 44 falsos alarmes") foram **removidos** de todos os
   documentos.
4. Acrescentada ressalva textual explícita: qualquer limiar localizado por busca no teste é **ilustrativo
   do formato da curva**, não valor a adotar em produção sem validação em dados novos.

**Resultado da comparação simétrica:**

| Detectados | Recall | XGBoost (FA) | Reg. Logística (FA) | Vantagem XGB |
|---:|---:|---:|---:|---:|
| 255 | 0,8979 | 23 | 25 | +2 |
| 265 | 0,9331 | 35 | 40 | +5 |
| 270 | 0,9507 | 52 | 63 | +11 |
| 272 | 0,9577 | 58 | 63 | +5 |

**Mudou alguma conclusão?** Sim, e para melhor: a vantagem do XGBoost **persiste** com a Regressão
Logística também otimizada — evidência mais forte que a anterior, e agora obtida sem seleção pós-hoc de um
único ponto.

**Onde:** `docs/04-modelagem.md` §9.5 · `RELATORIO-CONSOLIDADO.md` §6.6 · `03_treinamento_modelos.ipynb` §9

---

### Ajuste 3 · PR-AUC usado como "prova de dominância" sem teste

**O que estava errado.** A diferença de PR-AUC (0,9741 × 0,9710 = **0,0031**) é **menor** que a diferença
de F1 (0,012) que o próprio *bootstrap* já havia mostrado estar dentro do intervalo de confiança. Afirmar
que "a curva do XGBoost domina" sem testar significância era inconsistente com o padrão do resto do
documento.

**O que foi corrigido.** Calculado o intervalo de confiança do PR-AUC nas **mesmas 1.000 reamostragens**:

| Modelo | PR-AUC | IC 95% |
|---|---:|---|
| Regressão Logística | 0,9710 | [0,9587 — 0,9816] |
| Random Forest | 0,9697 | [0,9567 — 0,9811] |
| XGBoost | 0,9741 | [0,9623 — 0,9846] |

Diferença **pareada** XGBoost − Reg. Logística: **+0,0031**, IC 95% **[−0,0027, +0,0093]** — **contém o
zero**. P(XGBoost > Reg. Logística) = 85,4% das reamostragens.

**Mudou alguma conclusão?** A palavra "domina" foi **removida**. O PR-AUC passou de "evidência de
superioridade" a "consistente com uma leve vantagem, sem estabelecê-la". Isso **não invalida** seu uso como
critério de desempate — um desempate precisa ser objetivo e declarado de antemão, não estatisticamente
significativo.

**Onde:** `docs/10-significancia.md` §2.3 · `RELATORIO-CONSOLIDADO.md` §10 · `07_significancia.ipynb` §2.3

---

### Ajuste 4 · Escopo do McNemar não declarado

**O que estava errado.** O teste de McNemar comparou os modelos **no limiar padrão de 0,50 para ambos**,
mas o documento apresentava o resultado de significância como propriedade dos modelos, sem dizer que ele é
condicional àquele ponto de operação.

**O que foi corrigido.** Adicionada ressalva explícita: no limiar 0,50 os dois modelos operam em **níveis
de recall diferentes** (0,933 e 0,898); o teste não impede que o recall seja igualado ajustando o limiar de
um deles.

**Mudou alguma conclusão?** Sim, e é a correção mais substantiva desta rodada. A formulação passou de
*"a Regressão Logística detecta significativamente mais evadidos"* para:

> A diferença de recall **no limiar padrão** é real e não fruto do acaso. Mas ela é propriedade **do ponto
> de operação**, não do algoritmo — igualados os pontos, o XGBoost detecta mais evadidos pelo mesmo custo
> em 31 de 31 pontos da faixa prática.

*(Fraseado refinado na rodada 3: "faixa prática" foi substituído por "faixa de recall 0,852–0,9577", e
acrescentou-se a contagem de 72 dos 76 pontos decididos abaixo do teto de 50%.)*

**Onde:** `docs/10-significancia.md` §3.3 · `RELATORIO-CONSOLIDADO.md` §10.2 · `07_significancia.ipynb` §3.2

---

### Ajuste 5 · Matriz de confusão reconstruída por fórmula

**O que estava errado.** A tabela que compara a semente 42 contra as dez sementes reconstruía os valores
de VP/FN/FP **algebricamente**, a partir de recall e precisão, supondo `n_teste = 726` e
`n_positivos = 284` constantes em todas as sementes. **A suposição não havia sido verificada**, e
reconstruir inteiros a partir de duas métricas arredondadas introduz risco de erro silencioso.

**O que foi corrigido.** As dez sementes foram re-executadas capturando `confusion_matrix` **diretamente**.
A verificação passou a ser explícita na saída do notebook:

```
valores distintos de n_teste     : [726]
valores distintos de n_positivos : [284]
-> CONSTANTES
```

**Mudou alguma conclusão?** Não. A reconstrução algébrica estava **exata** — zero divergências em 33
comparações (11 sementes × 3 modelos). Mas o ponto metodológico procede: o que era suposição virou
verificação, e os números finais vêm de cálculo direto.

**Onde:** `07_significancia.ipynb` §4 · `dados/variabilidade_sementes.csv` (colunas VP/FN/FP/VN acrescentadas)

---

### Ajuste 6 · Linguagem sobre "precisão robusta"

**O que estava errado.** A frase *"precisão é robusta e não precisa de ressalva"* generalizava um achado
desta amostra como se fosse propriedade estrutural do modelo. Precisão e recall são interdependentes pelo
mesmo limiar; não há garantia de que a precisão seja sempre mais estável entre partições.

**O que foi corrigido.** Substituída por formulação que descreve o achado sem generalizar:

> Nesta verificação específica, a precisão **não apresentou** o mesmo viés otimista observado no recall — a
> semente 42 (0,9140) fica inclusive um pouco abaixo da média das dez (0,9191). Isso é um achado desta
> amostra, **não uma propriedade estrutural do modelo**.

**Mudou alguma conclusão?** Não. Apenas o grau de generalidade da afirmação.

**Onde:** `docs/10-significancia.md` §4.6 · `RELATORIO-CONSOLIDADO.md` §6.4 · `07_significancia.ipynb` §4.5

---

### Ajuste 7 · Tabela consolidada de semente

**O que estava errado.** Os números operacionais que vão para o resumo e a conclusão — *"de cada 100 alunos
que evadem, o modelo identifica 90"*, *"29 evadidos não detectados"* — vinham da semente 42 **sem ressalva
no mesmo parágrafo**. A descoberta do viés de semente ficava isolada numa seção técnica.

**O que foi corrigido.** Criada a tabela de duas versões, com as matrizes de confusão **reais**:

| Métrica | Semente 42 (reportada) | Média das 10 sementes | Faixa |
|---|---:|---:|---|
| F1 | 0,9059 | 0,8796 | 0,8656 – 0,8909 |
| Recall | 0,8979 | **0,8437** | 0,8239 – 0,8592 |
| Precisão | 0,9140 | 0,9191 | 0,8873 – 0,9412 |
| **Detectados de cada 100 evadidos** | **90** | **84** | 82 – 86 |
| **Acerto de cada 100 sinalizados** | **91** | **92** | 89 – 94 |
| Verdadeiros positivos | 255 | 240 | 234 – 244 |
| **Falsos negativos** | **29** | **44** | 40 – 50 |
| Falsos positivos | 24 | 21 | 15 – 31 |
| Alunos sinalizados | 279 | 261 | 253 – 275 |

E a ressalva foi propagada para **todos** os lugares onde os números aparecem — ver §"Onde os números
foram atualizados", abaixo.

**Mudou alguma conclusão?** Não a conclusão, mas **a precisão da comunicação**. O trabalho passa a reportar
84 como valor típico ao lado dos 90 da partição usada. Um refinamento apareceu no processo: **só o recall
tem viés otimista** — a precisão não.

**Onde:** `docs/10-significancia.md` §4.5 · `docs/06-resultados-e-figuras.md` Tabela 28 ·
`RELATORIO-CONSOLIDADO.md` §6.4 · `07_significancia.ipynb` §4.5

---

## Rodada 2 — três correções

### Correção 1 · Nota de não linearidade estendida a recall e precisão

**O que estava errado.** A nota explicava por que "F1 das médias" (0,8798) difere de "média dos F1"
(0,8796), mas **a mesma explicação se aplica a recall e precisão** e não estava documentada: 240/284 =
0,8451 contra os 0,8437 relatados; 240/261 = 0,9195 contra os 0,9191.

**O que foi corrigido.** A nota passou a cobrir as três métricas, com a tabela de comparação:

| Métrica | Média das métricas individuais | Recalculada das contagens médias |
|---|---:|---:|
| F1 | 0,8796 | 0,8798 |
| Recall | 0,8437 | 240 / 284 = 0,8451 |
| Precisão | 0,9191 | 240 / 261 = 0,9195 |

E deixou explícito que a coluna "média das 10 sementes" reporta a **média das métricas calculadas em cada
execução**, não a métrica recalculada das contagens médias — quantidades matematicamente diferentes porque
todas as três são razões, e a média de razões não é a razão das médias.

**Mudou alguma conclusão?** Não. Divergências de 0,001 a 0,002 são esperadas, não erro.

**Onde:** `docs/10-significancia.md` §4.5 · `docs/06-resultados-e-figuras.md` Tabela 28 · `07_significancia.ipynb` §4.5

---

### Correção 2 · "Quem prioriza recall deve preferir a Regressão Logística"

**O que estava errado.** A síntese afirmava que *"uma instituição que priorize recall acima de tudo opera
no regime onde a Regressão Logística é melhor"*. Impreciso: o XGBoost é **melhor ou igual em todo o
intervalo de recall 0,850 a 0,9577**. Mesmo uma política bastante agressiva — recall de 0,95 — ainda
favorece o XGBoost.

**O que foi corrigido.** A conclusão passou a localizar a exceção com precisão em vez de generalizar:

> Apenas políticas que demandam recall **próximo do teto** favorecem a Regressão Logística, numa banda
> estreita entre 0,9613 e 0,9718 que exige convocar praticamente metade da turma. Qualquer prioridade de
> recall abaixo desse ponto — incluindo políticas agressivas como recall de 0,95 — ainda favorece o
> XGBoost. **A exceção é estreita e está localizada com precisão, não é uma generalização.**

Isso **fortalece** a justificativa de manter o XGBoost: deixa de ser "F1 foi declarado a priori" e passa a
ser "o XGBoost é melhor ou equivalente na esmagadora maioria do espectro de política razoável; a exceção é
estreita, localizada e documentada".

**Mudou alguma conclusão?** Sim — a conclusão anterior era **generosa demais** com a Regressão Logística.
A corrigida é mais favorável ao modelo escolhido, e mais precisa.

**Onde:** `docs/04-modelagem.md` §9.5–9.6 · `docs/10-significancia.md` §3.3 ·
`RELATORIO-CONSOLIDADO.md` §6.6 e §12.3 item 6 · `docs/08-interpretabilidade-shap.md` R-7

---

### Correção 3 · Cruzamento localizado por interpolação grosseira

**O que estava errado.** O cruzamento entre as curvas foi localizado por interpolação em recall ≈ 0,956,
num trecho onde a grade de limiares era grosseira — os falsos alarmes saltavam de 52 para 113 entre pontos
vizinhos. Reportar isso apenas com a ressalva "a interpolação é frágil aqui" era insuficiente.

**O que foi corrigido — e o resultado é diferente.**

Primeiro, uma observação que mudou a abordagem: com 284 evadidos, **o recall só pode assumir múltiplos de
1/284 = 0,0035**. A resolução de 0,001 pedida é matematicamente inalcançável.

Em vez de refinar a grade de limiares, calculou-se a **curva exata**: ordenando os alunos pelo escore, para
cada número de evadidos detectados obtém-se o mínimo de falsos alarmes necessário. Zero interpolação,
resolução máxima possível — **um aluno**.

**O que a curva exata revelou:** o cruzamento único em 0,956 era **artefato da interpolação**. Na
realidade há **três inversões**, e a estrutura tem três regiões:

| Região | Recall | Pontos | Melhor modelo | Convocados |
|---|---|---:|---|---|
| **Estável** | 0,852 – 0,9577 | 31 | **XGBoost em 31 de 31** | 34% a 46% da turma |
| **Banda estreita** | 0,9613 – 0,9718 | 4 | **Reg. Logística em 4 de 4** | 47% a 49% da turma |
| **Instável** | ≥ 0,9754 | 8 | oscila — 3 inversões | 59% a 94% da turma |

> **Nota da rodada 3.** Esta tabela é mantida como registro histórico da rodada 2. Os intervalos de
> convocação estão corretos — consideram os dois modelos; pelo XGBoost isoladamente seriam 34%–45%,
> 47%–49% e 59%–91%. Dois pontos foram corrigidos depois: as **9** inversões pontuais não devem ser
> confundidas com as **3 travessias de liderança**, e, sobretudo, a leitura de que a banda estreita
> ficaria fora da faixa prática estava **errada** — ela convoca 47% a 49%, abaixo do teto de 50%.
> Ver a rodada 3 adiante.

As três inversões, para verificação:

| Detectados | Recall | XGBoost (FA) | Reg. Logística (FA) | Passa a favorecer |
|---:|---:|---:|---:|---|
| 273 | 0,9613 | 70 | 69 | Reg. Logística |
| 277 | 0,9754 | 154 | 163 | XGBoost |
| 279 | 0,9824 | 203 | 181 | Reg. Logística |

**Figura acrescentada:** `figuras/25_curvas_custo_pareadas.png` — painel esquerdo com a curva completa,
painel direito com zoom na região de decisão em **escala logarítmica** (necessária: o salto de 58 para 543
falsos alarmes comprimiria toda a faixa densa em escala linear), marcando as três inversões e o **teto de
convocação de 50% da turma**.

O teto transforma a figura de "aqui as curvas oscilam" em "aqui as curvas oscilam, e está fora da região
que qualquer coordenação conseguiria operar". Justificativa da cota: uma lista que sinaliza mais da metade
da turma deixa de ser priorização. É limite conservador e ilustrativo; o real depende da capacidade da
equipe de cada instituição.

**Mudou alguma conclusão?** Sim, duas vezes:

1. **De uma inversão para três**, e de "cruzamento em 0,956" para "banda estreita de quatro pontos onde a
   Regressão Logística ganha consistentemente, seguida de região instável".
2. **Uma afirmação minha anterior estava errada** e foi corrigida: eu havia dito que "não existe região em
   que a Regressão Logística seja confiavelmente melhor". **Existe** — a banda de recall 0,9613 a 0,9718,
   quatro pontos consecutivos. Não é oscilação ali; é vantagem real, num regime no limite da viabilidade
   operacional.

**Onde:** `docs/04-modelagem.md` §9.5 · `RELATORIO-CONSOLIDADO.md` §6.6 ·
`03_treinamento_modelos.ipynb` §9 · `figuras/25_curvas_custo_pareadas.png`

---

### Achado adicional, que emergiu da Correção 3

Um resultado apareceu na tabela isocusto e **merece destaque, não rodapé**, porque é evidência
**independente de todas as outras** — não precisa comparar os dois modelos entre si:

> No limiar padrão de 0,50, a Regressão Logística detecta 265 evadidos pagando **44** falsos alarmes. Mas a
> **sua própria** curva de custo mostra que os mesmos 265 evadidos são alcançáveis com apenas **40**.
>
> **O limiar padrão desperdiça 4 convocações do próprio modelo, sem ganho algum em detecção.**

Isso demonstra que **0,50 é convenção de biblioteca e não ponto de operação de nenhum dos dois modelos** —
a versão mais econômica do argumento central: o limiar importa mais que o algoritmo.

**Onde:** `docs/04-modelagem.md` §9.3 · `RELATORIO-CONSOLIDADO.md` §6.6 ·
`docs/08-interpretabilidade-shap.md` R-7 · `03_treinamento_modelos.ipynb` §9.3

---

## Onde os números foram atualizados com a ressalva de semente

| Arquivo | Local | O que mudou |
|---|---|---|
| `README.md` | Bloco de resultado principal | Acrescentada ressalva de partição com os valores típicos |
| `docs/00-visao-geral.md` | §8 — Estado atual e resultado principal | Idem |
| `docs/04-modelagem.md` | §11 — Tradução da matriz de confusão | "90 nesta partição" + valores típicos + ressalva sobre os quatro quadrantes |
| `docs/06-resultados-e-figuras.md` | Tabela 16 e §9 — Números-chave | Ressalva na leitura operacional; nova Tabela 28 com as duas versões |
| `docs/10-significancia.md` | §4.5 e §4.6 (novas) | Tabela completa das duas versões + orientação de redação |
| `RELATORIO-CONSOLIDADO.md` | §6.4, §12.1, §12.3 item 7 | Ressalva de partição nos três locais |

**A distinção que emergiu:** só o **recall** tem viés otimista (90 contra 84 típico). A **precisão** não —
a semente 42 (91) fica abaixo da média das dez (92). Reportar os dois como igualmente otimistas seria
impreciso.

---

## Confirmação final

### Nenhuma conclusão qualitativa mudou

| Conclusão | Situação |
|---|---|
| Modelo final: **XGBoost** | Inalterada — e melhor justificada |
| O sinal está nos dados, não no algoritmo | **Reforçada** — os ICs se sobrepõem, o PR-AUC não é significativo, a amplitude entre modelos é menor que o ruído da partição |
| Fatores de risco: desempenho acadêmico (59,9%) + situação financeira (14,9%) | Inalterada |
| Gatilhos operacionais (≤ 3 aprovações no 2º sem., ≥ 23 anos) | Inalterados |
| Três variáveis com efeito condicional oposto ao marginal | Inalterada |
| Teto intrínseco dos 29 falsos negativos | Inalterada (44 típicos entre partições) |
| Nenhuma disparidade de equidade significativa | Inalterada |
| A saída é um **escore de risco**, não probabilidade calibrada | Inalterada |

### O que mudou

1. **A contradição F1 × custo assimétrico foi resolvida**, não apenas apontada — com a curva de custo
   exata e a hierarquia de decisão em três níveis.
2. **Três evidências foram corrigidas** por erro de comparação (isocusto), assimetria com seleção pós-hoc
   (curva da Regressão Logística) e falta de teste de significância (PR-AUC).
3. **Os números operacionais passaram a ser reportados em duas versões**, com a ressalva no mesmo parágrafo.
4. **Duas afirmações minhas foram corrigidas**: que "a vantagem da Regressão Logística é artefato do limiar
   0,50" (é real naquele ponto) e que "não existe região em que ela seja confiavelmente melhor" (existe uma
   banda estreita).

### Estado do `RELATORIO-CONSOLIDADO.md`

**É a versão final antes da escrita da monografia.** Verificado:

- Nenhum resquício dos números antigos das Evidências A, B e C (limiar 0,43, "35 contra 44", "+17 vs +10",
  "domina");
- Todos os números operacionais com ressalva de partição;
- Todas as referências a arquivos, figuras e âncoras internas resolvem;
- Contagens de documentos, figuras e tabelas atualizadas.

## Rodada 3 — três pontos levantados em revisão externa

Revisão independente questionou três afirmações. Uma se mostrou incorreta, uma estava correta e passou
a ser reportada com mais precisão, e uma não tinha erro algum. Todas foram verificadas contra os dados,
não contra a documentação.

### Ponto 1 · A banda estreita estava *dentro* do teto prático — **erro real**

**O que estava escrito** (doc 04, §9.5 e relatório consolidado):

> "o XGBoost vence em 31 de 31 pontos de operação na faixa prática (recall ≤ 0,958); acima disso a
> ordenação oscila (3 inversões) e a operação deixa de ser útil, convocando de 45% a 94% da turma"

**O que os dados mostram.** Os quatro pontos em que a Regressão Logística vence (273 a 276 detectados)
convocam de **47,2% a 49,3%** da turma — **abaixo do teto de 50%** adotado na mesma seção. Não estão
fora da faixa prática; são os últimos quatro pontos antes da fronteira. A frase também contradizia o
parágrafo seguinte do próprio documento, que já dizia *"Não é oscilação ali; é uma vantagem real"*.

**Contabilidade correta, abaixo do teto de 50%:**

| | Pontos |
|---|---:|
| Decididos | 76 |
| XGBoost melhor | **72** |
| Regressão Logística melhor | **4** |
| Empatados | 200 |

**A escolha do teto não muda a conclusão.** A revisão sugeriu que a fronteira documentada pudesse ser
62% em vez de 50%. Testado: com 62% a contagem passa a 74 × 4 — **as mesmas quatro inversões**, porque
o segundo bloco (279 a 283) só começa a 66% de convocação. A hipótese não se confirmou, mas a
verificação expôs o erro real.

**Terceira imprecisão corrigida:** a curva tem **9 pontos** em que a Regressão Logística vence, não 3.
O número 3 conta *travessias de liderança*. Ambas as contagens são válidas; o texto passou a declarar
qual está usando.

**O "31 de 31" permanece válido** — recall 0,852 a 0,9577 são exatamente 31 pontos, todos do XGBoost.
O que mudou foi o enquadramento, não o número.

### Ponto 2 · O empate Random Forest × XGBoost — **nenhum erro**

A revisão pediu conferência célula a célula das matrizes de confusão, por considerar improvável o
empate em quatro casas decimais. Conferido: **TN 418, FP 24, FN 29, VP 255 nos dois modelos**,
idênticos. Não é erro de transcrição.

Acrescentado ao doc 04, §8, o detalhe que torna o mecanismo concreto: dos 24 alunos em que os modelos
discordam, **o Random Forest acerta 12 e o XGBoost acerta os outros 12** — compensação exata. As
probabilidades diferem (correlação 0,9712; diferença média absoluta 0,0768), confirmando que são
modelos distintos com o mesmo placar.

### Ponto 3 · A recalibração — **medição correta, redação imprecisa**

A revisão questionou se o `CalibratedClassifierCV` havia sido de fato treinado ou se a conclusão era
suposição. **Foi treinado** — sigmoid e isotonic, `cv=5` sobre o treino sem SMOTE, com Brier e ECE
tabelados antes e depois desde a primeira versão do documento.

Mas a revisão acertou no ponto fino: chamar o ganho de **"desprezível"** era impreciso. O isotonic
reduz o ECE de 0,1052 para 0,0880 — **16% em termos relativos**.

Executado o teste que faltava, sobre a faixa que motivou a análise ([0,10–0,20), 81 alunos, evasão real
de 2,5%):

| Método | Escore médio na faixa | Erro |
|---|---:|---:|
| sem calibração | 14,2% | +11,8 pp |
| sigmoid | 10,9% | +8,4 pp |
| isotonic | 13,2% | +10,7 pp |

**Nenhum corrige a faixa.** A afirmação "a calibração não resolve o problema real" ficou confirmada.

A justificativa foi reordenada: o argumento decisivo passa a ser a **monotonicidade** — o ROC-AUC é
idêntico nos três casos, logo a calibração não altera a ordenação e **exatamente os mesmos alunos
seriam convocados**. Muda o número colado no aluno, não a decisão.

### Rodada 3b · Varredura textual — exigida pela própria revisão

A revisão observou que era a terceira vez que uma afirmação sobre "quantas inversões e onde" precisava
ser corrigida, e que a cada vez ela reaparecia em lugares diferentes com fraseados ligeiramente
distintos. Exigiu, com razão, **varredura textual por grep** em vez de correção por memória.

A varredura buscou, em todos os arquivos `.md` e `.ipynb` do repositório, os termos: *inversão*,
*inversões*, *travessia*, *faixa prática*, *fora da faixa*, *banda estreita*, *31 de 31*, *62%*, *94%*,
*46%*, *75%*.

**Ela encontrou duas coisas que a correção por memória tinha deixado passar:**

1. **O notebook `03_treinamento_modelos.ipynb` ainda continha o texto original inteiro** — a tabela das
   três regiões, a manchete errada e a legenda "inversão de ordenação" na figura 25. Corrigir a
   documentação sem corrigir o notebook que a originou teria deixado a fonte contradizendo o derivado.
   O notebook foi corrigido e **reexecutado**; a figura 25 foi regerada com a legenda
   "travessia de liderança".

2. **Um erro introduzido pela própria correção da rodada 3.** Ao "corrigir" os intervalos de convocação
   de 34%–46% para 34%–45% e de 59%–94% para 59%–91%, houve troca silenciosa de convenção: os valores
   originais consideram **os dois modelos** (mínimo e máximo entre XGBoost e Regressão Logística),
   enquanto os novos consideravam apenas o XGBoost. Os originais estavam certos. Foram **revertidos**, e
   a convenção passou a ser declarada explicitamente sob cada tabela. A conclusão sobre o teto vale nas
   duas convenções — a banda estreita chega no máximo a 49,3%.

**Sobre os dois tetos.** O valor de 62% foi mantido onde é fato — a convocação resultante do limiar 0,10
— mas subordinado explicitamente ao teto de 50% em `docs/04` §10 e em `docs/06`. Não há mais duas
fronteiras concorrentes no corpo do texto: a fronteira é 50%, e 62% é uma consequência que está acima
dela.

**Reprodutibilidade conferida na reexecução:** das 21 colunas de `dados/metricas_modelos.csv`, 20 saíram
byte-idênticas; apenas o tempo de treino variou, por ser medida de relógio. Todos os arquivos `.joblib`
e todas as demais figuras saíram idênticos — inclusive `melhor_modelo.joblib`, o que confirma que a
restauração por cópia feita anteriormente era equivalente à geração nativa.

### Rodada 3c · Ressalva de partição nas células markdown dos notebooks

A revisão observou que a varredura anterior buscava em arquivos `.md` e por isso **não enxergava o texto
dentro de `"cell_type": "markdown"` nos `.ipynb`** — e que se um notebook havia escapado, outros
provavelmente também. Correto.

Nova varredura, agora extraindo as células markdown dos 9 notebooks e checando, **parágrafo a
parágrafo**, se cada número operacional da semente 42 tinha a ressalva junto. Resultado inicial:
**11 parágrafos sem ressalva**, em quatro notebooks — não apenas no `05_relatorio_final.ipynb`.

| Notebook | Parágrafos corrigidos |
|---|---|
| `03_treinamento_modelos.ipynb` | Tabela comparativa no limiar 0,50; matriz do empate RF × XGBoost |
| `04_interpretabilidade_shap.ipynb` | As três menções aos "29 falsos negativos" |
| `05_relatorio_final.ipynb` | Tabela-manchete, conclusão sobre o objetivo geral, L-01, teto intrínseco |
| `06_calibracao.ipynb` | Tabela de afirmações que dependem de calibração |

**Forma da correção.** Onde havia frase, o valor típico entrou ao lado do de semente 42 no mesmo
período. Onde havia tabela, o valor típico virou **coluna da própria tabela** ou o cabeçalho passou a
declarar a partição — para que número e qualificador continuem juntos mesmo se a tabela for copiada
isolada para a monografia. Nada foi para nota de rodapé nem para seção separada.

Exemplo, a tabela-manchete do `05_relatorio_final.ipynb`:

| Métrica | Partição avaliada (semente 42) | Partição típica (10 sementes) |
|---|---|---|
| **F1-score (Dropout)** | **0,9059** | **≈ 0,88** (0,866 – 0,891) |
| **Recall** | **0,8979** — 90 de cada 100 que evadem | **0,8437** — **84** de cada 100 (82 – 86) |
| **Precisão** | **0,9140** — 91 de cada 100 sinalizados | **0,9191** — 92 de cada 100, **sem viés** |

**Um falso positivo identificado e descartado:** o `06_calibracao.ipynb` cita "uma faixa com 279
alunos", que é o tamanho da faixa de calibração [0,0–0,1), não os 279 alunos sinalizados no limiar
0,50. Números iguais, grandezas diferentes.

### Rodada 3d · A mesma varredura por parágrafo nos documentos

Estendida a varredura por parágrafo aos `.md`: **69 parágrafos** com número operacional, dos quais
**38 isolados** (sem ressalva no parágrafo, no anterior ou no seguinte). Aplicar a ressalva a todos os
38 tornaria os documentos ilegíveis — a maioria é tabela de detalhe analítico (curvas de limiar,
intervalos de *bootstrap*, faixas de calibração, curva de custo) dentro de seções cujo assunto **é**
aquela partição.

A solução foi em dois níveis, e a distinção é deliberada:

| Nível | O que recebe | Forma |
|---|---|---|
| **1 — Tabelas-manchete** | O que um leitor cita como "o que o modelo alcança" | Valor típico **ao lado**, na própria tabela |
| **2 — Tabelas de detalhe** | Curvas, ICs, faixas, matrizes | **Aviso permanente no topo do documento** |

Documentos com aviso permanente: `docs/04`, `docs/06`, `docs/09`, `docs/12` e o
`RELATORIO-CONSOLIDADO.md`. Os demais (`docs/00`, `docs/05`, `docs/07`, `docs/10`, `README.md`) têm
ressalva pontual no parágrafo, porque citam poucos números operacionais.

**Cobertura final: 14 superfícies com números operacionais, 0 sem cobertura.**

Registre-se a divergência: a revisão pediu que nada ficasse "em nota de rodapé nem em seção separada".
Isso foi cumprido integralmente no nível 1. No nível 2, um aviso no topo do documento **é** uma forma
de declaração única — julgou-se que repetir a mesma ressalva em 38 tabelas de detalhe prejudicaria mais
a leitura do que ajudaria a precisão. A decisão está aqui para ser revertida se a orientadora preferir.

### Rodada 3e · Casos individuais — a ressalva que a de magnitude não cobre

A revisão distinguiu duas coisas que estavam sendo tratadas como uma. Corrigir 29 para 44 resolve a
**magnitude**. Não resolve o caso em que o notebook **caracteriza individualmente** quem são esses
alunos — porque o conjunto exato de falsos negativos muda de partição para partição, ainda que a
contagem coincida em ordem de grandeza.

**O cenário existe.** O Notebook 4 seleciona três casos por extremo (A: verdadeiro positivo de maior
probabilidade; B: verdadeiro negativo de menor; C: falso negativo mais confiante) e descreve o caso C
individualmente — "perfil dominado por fatores protetores" — para daí concluir que existe um teto
intrínseco. Era inferência a partir de **n = 1**.

**Duas correções foram feitas, não uma.**

**1. A caracterização virou achado agregado.** Nova célula no Notebook 4 compara o perfil dos 29 falsos
negativos com o dos evadidos detectados e o dos formandos:

| Variável | Falsos negativos | Evadidos detectados | Formandos |
|---|---:|---:|---:|
| Disciplinas aprovadas (2º sem.) | **6,24** | 1,23 | 6,10 |
| Média das notas (2º sem.) | **12,37** | 5,00 | 12,55 |
| Mensalidades em dia | **0,97** | 0,62 | 0,99 |

Em **5 das 7** variáveis-chave o falso negativo médio está mais próximo do formando. O caso C é retrato
do grupo, não exceção. Persistido em `dados/perfil_falsos_negativos.csv`.

Duas variáveis fogem: são **menos bolsistas** (7% contra 33%) e **mais velhos** (24,6 contra 21,6 anos)
— parecem formandos no desempenho e nas finanças, mas mantêm marcadores demográficos de risco.

**2. A ressalva específica dos casos individuais foi escrita.** Repetindo o pipeline em 10 sementes,
**216 alunos distintos** foram falso negativo em ao menos uma partição e **nenhum** nas dez. Não existe
um subconjunto de alunos intrinsecamente não detectáveis — existe uma região da fronteira cuja
atribuição depende da partição.

> **Superado pela rodada 3f.** A conclusão do parágrafo acima está **errada**. "Nenhum nas dez" é
> artefato da composição do split, não do modelo: com teste de 20%, um aluno aparece em ~2 das 10
> partições. Pela taxa **condicional**, 13,8% dos evadidos são errados em 100% das vezes em que são
> testados — existe, sim, núcleo duro. O parágrafo é mantido como registro histórico; a conclusão
> válida está na rodada 3f. O que é estável é o **perfil**, não a **lista**: nas 10 sementes o
grupo parece formando em 5 a 7 das 7 variáveis, com aprovações no 2º semestre entre 5,2 e 6,2.

**Consequência editorial:** a conclusão sobre o teto intrínseco permanece, agora apoiada no agregado. A
descrição do caso C passa a ser apresentada como *ilustração de um perfil*, nunca como "o aluno que o
modelo não detecta" — esse aluno, em outra partição, provavelmente seria detectado.

**Onde:** `04_interpretabilidade_shap.ipynb` (2 células novas, notebook reexecutado — todos os demais
artefatos SHAP saíram byte-idênticos), `docs/08`, `RELATORIO-CONSOLIDADO.md`.

### Rodada 3f · A conclusão da rodada 3e estava errada — existe núcleo duro

A rodada 3e afirmou que **não** existiria subconjunto intrinsecamente não detectável, apoiada em
"216 alunos foram falso negativo em ao menos uma das 10 sementes, e **nenhum** nas dez". A revisão
apontou o defeito: **com teste de 20%, um aluno aparece em cerca de 2 das 10 partições**, de modo que
"errado nas dez" é quase impossível por composição do split, não por comportamento do modelo. A
estatística estava confundida.

**A medida correta é condicional:** erros divididos pelas vezes em que o aluno foi efetivamente testado
como evadido. Aplicada, ela **inverte a conclusão**.

| Taxa condicional de falso negativo | Alunos | % |
|---|---:|---:|
| Nunca errado (0%) | 398 | 80,7% |
| Baixa (0 < t ≤ 30%) | 4 | 0,8% |
| Intermediária (30–99%) | 23 | 4,7% |
| **Sempre errado (100%)** | **68** | **13,8%** |

*Base: 493 evadidos com ao menos 3 aparições no teste ao longo das 10 partições.*

**Dois resultados sustentam a inversão:**

1. **A distribuição é bimodal.** 80,7% nunca errados, 13,8% sempre errados, menos de 6% no meio. Erro
   aleatório produziria concentração em torno da taxa média, não dois polos.
2. **O excesso sobre o acaso é enorme.** À taxa global de 15,6%, esperar-se-iam **1,3** alunos com taxa
   condicional de 100%. Observam-se **68** — cerca de **54 vezes mais**. Três apareceram no teste cinco
   vezes e não foram detectados em nenhuma.

**Frase que substitui a anterior**, adotada em `docs/08` e no `RELATORIO-CONSOLIDADO.md`:

> O erro do modelo **não é ruído distribuído entre os evadidos**: concentra-se num subconjunto
> identificável e estável. Existe um núcleo duro persistentemente não detectado, e é ele — não a
> variação amostral — que sustenta a afirmação de teto intrínseco. A afirmação fica **mais forte**, não
> mais fraca: o teto tem endereço.

O que **não** muda: os casos individuais A, B e C continuam sendo da partição da semente 42 e continuam
devendo ser apresentados como ilustração de perfil. O perfil agregado dos falsos negativos continua
estável entre partições (rodada 3e).

**Onde:** `04_interpretabilidade_shap.ipynb` (2 células novas; a célula 43 teve a afirmação confundida
substituída; notebook reexecutado sem divergência nos demais artefatos), `docs/08`,
`RELATORIO-CONSOLIDADO.md`, `dados/taxa_condicional_falsos_negativos.csv`.

**Observação registrada, deliberadamente não investigada.** Não se verificou se os 68 alunos do núcleo
duro compartilham o perfil "protetor" descrito na rodada 3e, nem o que os distingue dos demais falsos
negativos. É uma pergunta legítima e é candidata natural a trabalho futuro — mas esta era a última
verificação numérica antes da escrita, e abrir essa linha contrariaria o fechamento acordado.

### Rodada 3g · Auditoria de propagação da documentação

Perguntado se **todos** os documentos haviam sido atualizados, rodou-se uma auditoria cruzada: para
cada fato das rodadas 3 a 3f, verificou-se em quais dos 15 documentos ele aparecia. A resposta era
**não** — havia lacunas em quatro frentes, nenhuma delas de conteúdo errado, todas de conteúdo
faltante.

| Lacuna | O que faltava |
|---|---|
| **Contagens defasadas** | `docs/00`, `docs/04`, `docs/08` e `README` ainda diziam 44 e 49 células para os notebooks 3 e 4, que passaram a 50 e 53 após as reexecuções. O `README` dizia 29 figuras (são 31) e 31 tabelas (são 33) |
| **Notebook 9 sem documentação** | Não havia `docs/13`, e o notebook não constava do `README`, da árvore em `docs/00` nem do relatório consolidado |
| **Achados sem chegar às tabelas** | O perfil agregado dos falsos negativos e a taxa condicional não estavam em `docs/06`, que é a fonte de transcrição para a monografia |
| **Núcleo duro fora das limitações** | `docs/07` não registrava o achado como limitação nem o desdobramento como trabalho futuro |

**O que foi feito.** Criado o [`docs/13`](13-aplicacao-alunos-ativos.md); acrescentadas as Tabelas 32 e
33 e as figuras 26 e 27 ao [`docs/06`](06-resultados-e-figuras.md); criadas a limitação **L-01b** e o
trabalho futuro **TF-10** em [`docs/07`](07-limitacoes-e-trabalhos-futuros.md); acrescentada a seção
**11b** ao relatório consolidado, além do perfil agregado dos falsos negativos e do detalhe 12 × 12 do
empate; corrigidas todas as contagens.

**Verificação final:** 0 links quebrados entre os 15 documentos, e todos os artefatos novos
(`alunos_ativos_priorizados.csv`, `perfil_falsos_negativos.csv`,
`taxa_condicional_falsos_negativos.csv`, figuras 26 e 27) citados em ao menos dois documentos.

**Lição registrada.** Corrigir um achado tem duas metades: acertar o número e **propagá-lo**. Nas
rodadas 3 a 3f a primeira metade recebeu verificação sistemática; a segunda foi feita de memória, e a
memória falhou em quatro frentes. A auditoria cruzada deveria ter sido parte de cada rodada, não uma
etapa final.

### Fraseado canônico adotado

Toda menção ao assunto passou a usar esta formulação:

> Na faixa de recall 0,852–0,9577 o XGBoost vence em **31 de 31** pontos de operação. Abaixo do teto de
> convocação de 50%, considerando todos os pontos decididos, ele vence **72 de 76**; os 4 restantes
> pertencem à Regressão Logística e convocam de 47% a 49% da turma — **dentro** do teto, nos últimos
> pontos percentuais antes dele. A curva tem **9 pontos** vencidos pela Regressão Logística e
> **3 travessias de liderança**: são contagens distintas.

### Resumo da rodada 3

| Ponto | Veredito | Arquivos alterados |
|---|---|---|
| Faixa prática e banda estreita | **Erro corrigido** | doc 04, doc 05, doc 06, doc 10, relatório consolidado |
| Empate RF × XGBoost | Confirmado, detalhe acrescentado | doc 04 |
| Recalibração | Redação corrigida, conclusão mantida | doc 09, relatório consolidado |

---

### Artefatos acrescentados nesta revisão

| Arquivo | Conteúdo |
|---|---|
| `dados/curva_custo_exata.csv` | Curva de custo exata dos dois modelos, sem interpolação |
| `dados/numeros_operacionais_semente.csv` | Tabela das duas versões dos números operacionais |
| `figuras/25_curvas_custo_pareadas.png` | Curvas de custo pareadas, com zoom em escala log e teto de convocação |
| `docs/12-revisao-final.md` | Este changelog |
