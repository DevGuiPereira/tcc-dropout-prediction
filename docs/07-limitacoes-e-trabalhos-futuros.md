# 07 — Limitações e Trabalhos Futuros

> Consolidação honesta do que o trabalho **não** sustenta e por quê.
>
> Este documento não é uma formalidade. Uma limitação declarada e compreendida é um ponto forte na
> defesa; a mesma limitação descoberta pela banca é um ponto fraco. As limitações abaixo estão
> ordenadas por gravidade — as três primeiras são as que mais restringem as conclusões.

---

## Parte I — Limitações

### L-01: Diferença entre a população de treino e a população de aplicação

**A limitação mais séria do trabalho.**

Ao remover a classe `Enrolled` (decisão [D-09](05-decisoes-metodologicas.md)), o modelo passou a ser
treinado e validado **exclusivamente em trajetórias concluídas** — alunos que já se formaram ou já
evadiram. Mas o uso pretendido é o oposto: em produção, o modelo seria aplicado a **alunos ativos**,
que é precisamente a população excluída do treino.

| | População de treino | População de aplicação |
|---|---|---|
| Quem são | 3.630 alunos com desfecho definido | Alunos matriculados, sem desfecho |
| No dataset | `Dropout` + `Graduate` | `Enrolled` (794 alunos, 17,9%) |

**Por que isso importa.** Não é apenas uma questão de tamanho de amostra. Alunos ativos podem diferir
sistematicamente dos que já concluíram sua trajetória — trancamentos, reprovações acumuladas e
mudanças de curso produzem perfis que o treino não contém. O desempenho de 0,906 de F1 — que é o da
partição da semente 42; em partições típicas, ≈ 0,88 pela [Verificação 2](10-significancia.md) — é uma estimativa
válida **para a população de trajetórias concluídas**, e não há garantia de que se mantenha na população
de aplicação.

**O que foi feito a respeito.** A formulação multiclasse está implementada e ativável em uma linha,
permitindo reportá-la como análise de sensibilidade. A limitação está registrada em três lugares
(notebook, documentação de decisões e aqui).

**O que seria necessário para resolver.** Modelagem de sobrevivência com dados censurados — ver
[TF-01](#tf-01-modelagem-de-sobrevivência-com-dados-censurados).

---

### L-01b: Um núcleo duro de evadidos que o modelo não detecta

Entre os 493 evadidos que apareceram no conjunto de teste ao menos 3 vezes nas 10 partições,
**68 (13,8%) foram classificados errado em 100% das vezes**. A distribuição da taxa condicional de erro
é **bimodal** — 80,7% nunca errados, 13,8% sempre errados, menos de 6% no meio —, e erro aleatório à
taxa global de 15,6% produziria apenas 1,3 alunos nessa condição.

**Por que isso importa.** O erro do modelo não é ruído distribuído entre os evadidos: concentra-se num
subconjunto identificável e estável. Cerca de um em cada sete evadidos é sistematicamente invisível
para este conjunto de variáveis, e nenhuma escolha de algoritmo ou de limiar o alcança.

**O que foi feito a respeito.** O achado está medido e documentado ([doc 08](08-interpretabilidade-shap.md)),
com o artefato `dados/taxa_condicional_falsos_negativos.csv`. Ele **fortalece** a afirmação de teto
intrínseco, que antes se apoiava em um único caso ilustrativo.

**O que seria necessário para resolver.** Ampliar o conjunto de variáveis — ver
[TF-10](#tf-10-caracterizar-o-núcleo-duro-de-evadidos-não-detectados).

---

### L-02: Aplicabilidade prática restrita pelo momento da observação

As quatro variáveis mais preditivas do modelo são de desempenho do **1º e 2º semestres**:

| Variável | Importância RF |
|---|---:|
| Disciplinas aprovadas (2º sem.) | 17,8% |
| Disciplinas aprovadas (1º sem.) | 12,1% |
| Média das notas (2º sem.) | 10,8% |
| Média das notas (1º sem.) | 6,2% |
| **Somadas** | **46,9%** |

Essas variáveis **só existem ao final do primeiro ano de curso**. Um modelo que depende delas para
quase metade de sua capacidade preditiva:

- **não pode ser usado no ato da matrícula**, quando a intervenção seria mais barata;
- **identifica o aluno quando o processo de abandono já está avançado** — a mediana de disciplinas
  aprovadas no 2º semestre entre os evadidos é **zero**, ou seja, o aluno já parou de estudar quando o
  modelo o sinaliza.

**Tensão honesta:** parte do alto desempenho do modelo vem justamente de observar o aluno tarde. Um
modelo de alerta precoce, com apenas dados de matrícula, teria desempenho necessariamente inferior — mas
seria **mais útil**. Desempenho e utilidade não coincidem aqui, e essa é uma discussão que o trabalho
deve levantar em vez de esconder atrás do F1.

Ver [TF-02](#tf-02-modelo-de-alerta-precoce).

---

### L-03: Generalização geográfica e institucional

Os dados provêm de **uma única instituição portuguesa** (Instituto Politécnico de Portalegre), com
coleta de 2008 a 2019.

| Aspecto | Diferença relevante para o Brasil |
|---|---|
| Sistema de notas | Escala 0–20 (Portugal) × 0–10 ou 0–100 (Brasil) |
| Ingresso | Modalidades portuguesas × ENEM/SISU, vestibular, PROUNI, FIES |
| Financiamento | Estrutura de mensalidades e bolsas distinta |
| Contexto socioeconômico | Distribuição de renda, escolaridade parental e mobilidade diferentes |
| Codificação | Ocupações e qualificações seguem a classificação portuguesa |

**Consequência:** os **coeficientes e limiares específicos não são transferíveis**. Um modelo treinado
nesses dados não deve ser aplicado a uma instituição brasileira sem retreinamento.

**O que provavelmente se transfere** são os **padrões qualitativos** — a centralidade do desempenho
acadêmico inicial, o peso da situação financeira, o efeito protetor da bolsa. Esses achados são
consistentes com a literatura brasileira sobre evasão, o que sugere que o mecanismo é geral mesmo que os
parâmetros não sejam.

Ver [TF-03](#tf-03-validação-em-dados-brasileiros).

---

### L-04: Partição única de teste, sem intervalo de confiança

Todas as métricas finais vêm de **uma** divisão treino/teste com `random_state=42`. A validação
cruzada mitiga isso no treino, mas o número reportado (F1 = 0,9059) vem de uma partição só.

Com 726 registros de teste e 284 evadidos, **cada aluno reclassificado move o F1 em cerca de 0,0019** —
ou seja, uma diferença de 0,01 em F1 equivale a apenas **5 alunos**. Isso está dentro da faixa em que
outra partição poderia inverter o ranking dos modelos.

**Consequência prática:** o trabalho **não pode afirmar** que o XGBoost é superior aos demais. E, de
fato, não afirma: o texto trata os três modelos como equivalentes na prática, e o vencedor foi definido
por um critério de desempate declarado.

Ver [TF-04](#tf-04-quantificação-da-incerteza).

---

### L-05: Ausência de teste de significância estatística

As diferenças entre modelos não foram submetidas a teste (McNemar seria o adequado para comparar dois
classificadores no mesmo conjunto de teste).

**Decisão consciente de escopo.** Em compensação, nenhuma afirmação do trabalho depende de significância:
a conclusão central é que **os modelos são equivalentes**, e um teste que confirmasse equivalência
estatística **reforçaria** essa tese em vez de contrariá-la.

O empate exato entre Random Forest e XGBoost (matriz 418/24/29/255 na partição da semente 42) é, por si só, evidência forte de
equivalência prática.

---

### L-06: F1-score assume custos simétricos que o próprio trabalho nega

Aqui há uma inconsistência interna que precisa ser explicitada.

O trabalho afirma repetidamente que **falso negativo custa mais que falso positivo** — perder um aluno
é pior que uma conversa desnecessária. Mas o F1-score, adotado como critério de decisão, é a média
harmônica de precisão e recall: ele **pesa os dois erros igualmente**.

A escolha do F1 foi justificada por comparabilidade com a literatura, o que é legítimo. Mas a métrica
teoricamente correta seria:

- **F-beta com β > 1** (por exemplo, F2), que pondera recall acima de precisão; ou
- **perda esperada com custos explícitos**, que exigiria da instituição uma estimativa do custo real de
  perder um aluno (evasão custa em receita, em vaga ociosa e em indicador institucional).

**Mitigação adotada:** a análise de limiar (§10 do [doc 04](04-modelagem.md)) percorre todo o espectro
de trade-off e apresenta os cenários em número de alunos, permitindo que a decisão de custo seja tomada
pela instituição em vez de embutida na métrica.

---

### L-07: Validação cruzada não aninhada

Os hiperparâmetros foram escolhidos por validação cruzada de 5 dobras sobre o treino, e o desempenho
final foi medido no teste. O procedimento é correto e não vaza informação do teste.

Porém, a **estimativa de desempenho da CV** (F1 = 0,8748 para o XGBoost) é otimista, porque o mesmo
esquema de dobras foi usado para selecionar entre 243 configurações — a melhor delas foi escolhida em
parte pelo ruído favorável das dobras. A estimativa honesta do desempenho pós-seleção exigiria
**validação cruzada aninhada**.

**Atenuação:** o número reportado como resultado do trabalho é o do **conjunto de teste**, não o da CV,
e este não sofre do problema. A diferença observada entre CV e teste (+0,031) é pequena e positiva,
sugerindo que o efeito é modesto.

---

### L-08: SMOTE interpola sobre variáveis categóricas

> **Atualização após a Fase 5.** O impacto desta limitação ficou menor do que se temia. Das 15 variáveis
> mais importantes pelo SHAP, **nenhuma é uma dummy** — todas são contínuas ou binárias. As categóricas
> só aparecem no topo depois de recompostas, e mesmo assim `Curso` alcança apenas a 8ª posição (3,77%).
> Os valores fracionários gerados pelo SMOTE afetam, portanto, a parte menos influente do modelo.

O SMOTE clássico gera pontos sintéticos por interpolação linear, produzindo valores fracionários nas
78 *dummies* — por exemplo, 0,34 em `course_9500`, onde só existiriam 0 ou 1. Isso afeta os 630
registros sintéticos do treino.

`SMOTENC`, que trata colunas categóricas de forma discreta, seria mais rigoroso. Optou-se pelo SMOTE
clássico por comparabilidade com a literatura desta base ([D-16](05-decisoes-metodologicas.md)).

**Magnitude estimada do problema:** pequena. As dummies concentram parcela menor da importância total
(as 15 variáveis mais importantes são todas contínuas ou binárias) e os modelos de árvore lidam bem com
valores fracionários em variáveis indicadoras. Mas o efeito não foi quantificado.

---

### L-09: Dataset artificialmente limpo

Zero valores ausentes, zero duplicatas, zero inconsistências — resultado do pré-tratamento feito pelos
autores antes da publicação.

Um sistema acadêmico real apresenta notas faltantes, matrículas duplicadas, campos socioeconômicos em
branco e inconsistências de digitação. **O pipeline não foi exposto a nenhum desses problemas** e não
contém etapa de imputação ou deduplicação.

Ironicamente, o próprio projeto encontrou um problema de qualidade de dados grave — mas na **camada de
armazenamento** (3.230 notas corrompidas no MySQL), não nos dados de origem. Isso reforça o ponto: dados
reais falham, e o pipeline precisaria de defesas que hoje não tem.

Ver [doc 02](02-qualidade-e-integridade.md).

---

### L-10: Variáveis macroeconômicas como identificador de coorte

`unemployment_rate`, `inflation_rate` e `gdp` descrevem o contexto econômico do **ano de ingresso**, não
o indivíduo. Com poucos valores distintos, funcionam na prática como um **identificador aproximado da
coorte**.

O modelo pode usá-las para "descobrir o ano" e explorar diferenças de taxa de evasão entre coortes.
Isso é sinal real dentro da amostra, mas **não generaliza para anos futuros**, cujos valores
macroeconômicos serão diferentes de todos os observados no treino.

**Atenuação:** a importância medida dessas três variáveis é baixa (< 1,5% cada), então o efeito é
limitado. Mantidas porque removê-las sem evidência seria seleção arbitrária
([D-07](05-decisoes-metodologicas.md)).

---

### L-11: Ausência de análise de equidade algorítmica

> **Atualização após a Fase 5.** A interpretabilidade tornou esta limitação mais urgente, não menos. O
> SHAP confirmou que **gênero** contribui para as predições (ρ = +0,55 na direção de risco) e que
> variáveis socioeconômicas — bolsa, inadimplência, ocupação dos pais — estão entre os fatores ativos.
> Saber *que* o modelo usa esses atributos torna obrigatório verificar *se* ele erra mais para algum
> subgrupo.

O gênero aparece como fator de risco relevante (r = +0,204; código 1 associado a 45,1% de evasão contra
25,1%), e há variáveis diretamente ligadas a condição socioeconômica (escolaridade e ocupação dos pais,
bolsa, inadimplência).

O trabalho **não avaliou** se o modelo apresenta desempenho desigual entre subgrupos — se, por exemplo,
o recall é sistematicamente menor para mulheres, para alunos mais velhos ou para alunos de baixa renda.

**Por que isso é relevante além do rigor técnico.** Um sistema de alerta é usado para **alocar recursos
de apoio**. Se ele detecta pior um subgrupo, esse subgrupo recebe menos apoio — e o sistema reproduz ou
amplifica a desigualdade que deveria combater. Métricas como *equalized odds* e paridade de recall por
subgrupo existem exatamente para isso.

Esta é, na avaliação deste documento, **a limitação mais importante entre as não previstas no escopo
original** do trabalho.

Ver [TF-05](#tf-05-auditoria-de-equidade-algorítmica).

---

### L-12: Resultados condicionais a uma única semente aleatória

`random_state=42` em toda operação estocástica garante **reprodutibilidade**, mas não informa **estabilidade**.
Não se sabe quanto os resultados variariam com outras sementes — na partição, no SMOTE ou na
inicialização dos modelos.

Ver [TF-04](#tf-04-quantificação-da-incerteza).

---

### L-13: Categorias raras agrupadas dificultam a interpretação

O agrupamento com `min_frequency=0.01` reduziu 200 categorias a 78 dummies, criando grupos
`infrequent_sublevel` que **misturam categorias heterogêneas**. Por exemplo, `nacionality` teve 17 de
18 categorias agrupadas.

Consequência para a Fase 5: se uma dummy `infrequent_sublevel` aparecer com valor SHAP relevante, será
difícil dizer **qual** categoria específica causou o efeito. O trade-off foi aceito conscientemente
([D-14](05-decisoes-metodologicas.md)) porque a alternativa (200 dummies com dezenas de casos isolados)
seria pior tanto estatisticamente quanto para interpretação.

---

### L-14: SHAP explica o modelo, não o mundo

Todos os resultados de interpretabilidade da Fase 5 medem **como o XGBoost usa cada variável**, e não o
**efeito causal** da variável sobre a evasão.

A afirmação "inadimplência contribui com +1,86 em log-odds" é verdadeira e verificável. A afirmação
"quitar a dívida do aluno reduziria seu risco em X" **não decorre dela** e exigiria delineamento
experimental — atribuição aleatória de auxílio financeiro, com grupo de controle.

Três recomendações do trabalho são especialmente sensíveis a essa distinção:

| Recomendação | Problema causal |
|---|---|
| **R-3** (ampliar bolsas) | **Viés de seleção provável.** Bolsas costumam exigir desempenho mínimo, logo bolsistas já são um grupo selecionado. Parte da diferença de 12,2% × 38,7% pode refletir a seleção, não a bolsa |
| **R-2** (inadimplência como alerta) | A inadimplência pode ser **sintoma** da decisão de sair, não causa. O aluno que já decidiu abandonar naturalmente para de pagar |
| **R-4** (apoio a ingressantes mais velhos) | Idade é **marcador**, não causa. O que provavelmente opera são as responsabilidades concorrentes que a acompanham |

Confundir associação preditiva com efeito causal é o erro mais comum na aplicação de interpretabilidade
a políticas públicas — e o mais fácil de uma banca identificar. As recomendações do trabalho estão
redigidas como "onde intervir", não como "o que funciona".

Ver [doc 08](08-interpretabilidade-shap.md), §2 e §11.

---

### L-15: Ausência de validação temporal

O dataset cobre ingressos de 2008 a 2019, mas a divisão treino/teste foi **aleatória**, não temporal. O
modelo pode ter treinado com coortes de 2018 e sido testado com coortes de 2010.

Uma divisão temporal (treinar nos anos antigos, testar nos recentes) mediria a capacidade de
**extrapolação para o futuro**, que é o uso real. A validação aleatória tende a produzir estimativas
mais otimistas, porque o modelo vê exemplos contemporâneos ao teste.

---

## Parte II — Trabalhos Futuros

Ordenados por relação entre impacto e viabilidade.

### TF-01: Modelagem de sobrevivência com dados censurados

**Resolve:** [L-01](#l-01-diferença-entre-a-população-de-treino-e-a-população-de-aplicação)

Tratar `Enrolled` corretamente, como observação **censurada à direita**, em vez de descartá-la. Modelos
adequados: regressão de Cox, estimador de Kaplan-Meier, *random survival forests*.

**Ganhos:**
- Usa os 794 alunos hoje descartados (17,9% da amostra)
- Responde **quando** o aluno tende a evadir, não apenas **se** — informação diretamente útil para
  cronograma de intervenção
- Elimina a diferença entre população de treino e de aplicação

**Viabilidade:** alta. Requer a variável "tempo até o evento", derivável dos dados semestrais.
**Impacto:** o maior de todos — corrige a limitação central do trabalho.

---

### TF-02: Modelo de alerta precoce

**Resolve:** [L-02](#l-02-aplicabilidade-prática-restrita-pelo-momento-da-observação)

Treinar um segundo modelo usando **apenas** variáveis disponíveis no ato da matrícula: demográficas,
socioeconômicas, nota de admissão, modalidade de ingresso, ordem de opção do curso. Sem nenhuma variável
de desempenho semestral.

**Resultado esperado:** desempenho substancialmente inferior — as variáveis do 1º e 2º semestres somam
46,9% da importância. Mas o modelo seria aplicável **no primeiro dia de aula**.

**Desenho sugerido:** uma **arquitetura em dois estágios**, com o modelo de matrícula fazendo a triagem
inicial e o modelo completo refinando ao final de cada semestre. Comparar a curva de desempenho por
momento de observação (matrícula → fim do 1º sem. → fim do 2º sem.) produziria um resultado de interesse
prático direto: **quanto se ganha em precisão por cada semestre de espera, e quantos alunos se perde
nesse intervalo.**

**Viabilidade:** muito alta — os dados já estão prontos, basta restringir as colunas.
**Impacto:** alto. Transforma um exercício preditivo em ferramenta de gestão.

---

### TF-03: Validação em dados brasileiros

**Resolve:** [L-03](#l-03-generalização-geográfica-e-institucional)

Replicar o pipeline com dados de uma instituição brasileira — a própria UEMG seria o caso natural.

**Duas perguntas de pesquisa:**
1. Os **padrões qualitativos** se mantêm? (desempenho inicial e situação financeira como fatores
   dominantes)
2. Os **parâmetros** mudam muito? Quanto de retreinamento é necessário?

**Viabilidade:** média — depende de acesso a dados institucionais e aprovação em comitê de ética.
**Impacto:** alto. É o que converteria o trabalho em contribuição aplicada ao contexto local.

---

### TF-04: Quantificação da incerteza

**Resolve:** [L-04](#l-04-partição-única-de-teste-sem-intervalo-de-confiança),
[L-05](#l-05-ausência-de-teste-de-significância-estatística),
[L-07](#l-07-validação-cruzada-não-aninhada),
[L-12](#l-12-resultados-condicionais-a-uma-única-semente-aleatória)

Quatro procedimentos independentes, todos de baixo custo:

| Procedimento | O que entrega |
|---|---|
| *Bootstrap* do conjunto de teste (1.000 reamostragens) | Intervalos de confiança de 95% para todas as métricas |
| Repetição com múltiplas sementes (10–30 execuções) | Variabilidade atribuível à partição e ao SMOTE |
| Teste de McNemar entre pares de modelos | Se as diferenças são estatisticamente distinguíveis |
| Validação cruzada aninhada | Estimativa honesta do desempenho pós-seleção de hiperparâmetros |

**Viabilidade:** muito alta — o pipeline já está pronto, é só envolvê-lo em laços. Custo computacional
irrelevante (cada execução completa leva ~2 minutos).
**Impacto:** médio-alto. Não muda as conclusões, mas as torna defensáveis com rigor estatístico. **É o
melhor retorno por esforço da lista.**

---

### TF-05: Auditoria de equidade algorítmica

**Resolve:** [L-11](#l-11-ausência-de-análise-de-equidade-algorítmica)

Medir recall, precisão e taxa de falsos negativos **por subgrupo**: gênero, faixa de idade, condição de
bolsista, turno, escolaridade dos pais. Aplicar critérios de *equalized odds* e paridade de recall.

**A pergunta central:** o modelo detecta pior algum grupo? Se sim, esse grupo receberá menos apoio, e o
sistema amplificará a desigualdade que deveria combater.

**Viabilidade:** muito alta — basta segmentar as predições já produzidas.
**Impacto:** alto, e de natureza ética, não apenas técnica. Em um trabalho cujo produto é a alocação de
apoio institucional, esta análise é praticamente obrigatória.

---

### TF-06: Otimização com custos assimétricos explícitos

**Resolve:** [L-06](#l-06-f1-score-assume-custos-simétricos-que-o-próprio-trabalho-nega)

Substituir o F1 por uma função de perda que reflita os custos reais:

```
perda = C_FN × (evadidos não detectados) + C_FP × (falsos alarmes)
```

Requer da instituição uma estimativa de `C_FN` (receita perdida + vaga ociosa + impacto em indicadores)
e `C_FP` (custo de hora de atendimento). Com esses dois números, o limiar ótimo passa a ser **calculado**
em vez de escolhido.

**Viabilidade:** alta tecnicamente; depende de informação institucional.
**Impacto:** médio-alto. Alinharia a métrica ao objetivo declarado do trabalho.

---

### TF-07: Validação temporal

**Resolve:** [L-14](#l-15-ausência-de-validação-temporal)

Dividir treino e teste por **coorte de ingresso** em vez de aleatoriamente: treinar nos anos mais
antigos, testar nos mais recentes. Mede a capacidade de extrapolação para o futuro, que é o uso real do
modelo.

**Viabilidade:** média — exige identificar o ano de ingresso, possivelmente derivável das variáveis
macroeconômicas.
**Impacto:** médio. Provavelmente revelaria desempenho inferior ao reportado — o que seria um resultado
honesto e valioso.

---

### TF-08: Comparação SMOTE × SMOTENC × `class_weight`

**Resolve:** [L-08](#l-08-smote-interpola-sobre-variáveis-categóricas)

Executar o pipeline com as três estratégias de balanceamento e comparar. Custo: ~6 minutos.

**Viabilidade:** muito alta.
**Impacto:** baixo-médio. Provavelmente confirmaria equivalência prática, fechando a ressalva com
evidência em vez de argumento.

---

### TF-09: Protótipo de sistema de alerta

Transformar o modelo em ferramenta utilizável: interface web que recebe os dados de um aluno, retorna a
probabilidade de evasão e **a explicação SHAP dos fatores que mais contribuíram**, com sugestão de tipo
de intervenção conforme o fator dominante (financeiro → assistência estudantil; acadêmico → apoio
pedagógico).

**Viabilidade:** alta — `preprocessador.joblib` + `melhor_modelo.joblib` já bastam; Streamlit ou FastAPI
resolveriam a interface.
**Impacto:** alto em demonstração e em potencial de continuidade do trabalho.

---

### TF-10: Caracterizar o núcleo duro de evadidos não detectados

A [L-01b](#l-01b-um-núcleo-duro-de-evadidos-que-o-modelo-não-detecta) mostrou que 13,8% dos evadidos
são classificados errado em **100%** das vezes em que aparecem no teste. Sabe-se que existem e quantos
são; **não se sabe o que os distingue** dos demais falsos negativos nem dos evadidos que o modelo
detecta.

A investigação natural é comparar esses 68 alunos com os outros dois grupos nas 36 variáveis, e
verificar se compartilham o perfil "protetor" descrito no [doc 08](08-interpretabilidade-shap.md) ou se
formam um padrão próprio. Se houver padrão próprio, ele indica **quais variáveis faltam** ao conjunto
atual — que é a única via para elevar o teto intrínseco.

**Viabilidade:** alta — `dados/taxa_condicional_falsos_negativos.csv` já identifica os alunos; a
comparação é descritiva.
**Impacto:** alto — é o caminho direto para saber que dados a instituição precisaria coletar.

> Esta investigação foi **deliberadamente não realizada** neste trabalho. A verificação que revelou o
> núcleo duro foi a última da fase de análise, e abrir a caracterização exigiria uma nova rodada de
> testes — decisão registrada na rodada 3f do [changelog](12-revisao-final.md).

---

## Parte III — Recomendação de prioridades

Se houver tempo para apenas três ações antes da entrega:

| Prioridade | Ação | Justificativa |
|---|---|---|
| **1ª** | [TF-04](#tf-04-quantificação-da-incerteza) — *bootstrap* + McNemar | Custo quase zero, o pipeline já existe. Elimina de uma vez quatro limitações e blinda o trabalho contra a pergunta "isso é significativo?" |
| **2ª** | [TF-02](#tf-02-modelo-de-alerta-precoce) — modelo de matrícula | Os dados já estão prontos. Converte o trabalho de exercício preditivo em ferramenta de gestão, e a comparação "quanto se ganha por semestre de espera" é um resultado original |
| **3ª** | [TF-05](#tf-05-auditoria-de-equidade-algorítmica) — auditoria de equidade | Baixo custo, alto peso ético. Em um trabalho sobre alocação de apoio institucional, é a análise cuja ausência mais chama atenção. A Fase 5 tornou-a mais urgente: o SHAP confirmou que o modelo usa gênero e variáveis socioeconômicas |

As três somadas custam poucas horas de execução e endereçam seis das quinze limitações listadas.
