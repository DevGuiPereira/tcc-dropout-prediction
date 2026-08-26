# 12 — Revisão Final: Changelog das Correções

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

### Artefatos acrescentados nesta revisão

| Arquivo | Conteúdo |
|---|---|
| `dados/curva_custo_exata.csv` | Curva de custo exata dos dois modelos, sem interpolação |
| `dados/numeros_operacionais_semente.csv` | Tabela das duas versões dos números operacionais |
| `figuras/25_curvas_custo_pareadas.png` | Curvas de custo pareadas, com zoom em escala log e teto de convocação |
| `docs/12-revisao-final.md` | Este changelog |
