# 05 — Registro de Decisões Metodológicas

> Consolidação de **todas** as decisões tomadas no pipeline, cada uma com a alternativa rejeitada, a
> justificativa e o risco assumido.
>
> Este documento existe para a **defesa**. Toda pergunta do tipo "por que você fez X e não Y?" deve ter
> resposta aqui, com o local exato da implementação.

---

## Como ler este documento

Cada decisão tem cinco campos:

| Campo | Significado |
|---|---|
| **Decisão** | O que foi feito |
| **Alternativa rejeitada** | O que se poderia ter feito no lugar |
| **Justificativa** | Por que a escolha, com evidência quando houver |
| **Risco / limitação** | O que se perde ou o que pode dar errado |
| **Onde** | Arquivo e seção da implementação |

---

## Fase 2 — Entendimento dos Dados

### D-01 · Usar o MySQL local como fonte primária, com fallback para o UCI

| | |
|---|---|
| **Alternativa rejeitada** | Usar apenas o `ucimlrepo` (mais simples) ou apenas o MySQL (sem rede de segurança) |
| **Justificativa** | O banco dimensional é parte do trabalho do autor e demonstra competência em modelagem de dados. O fallback garante que o pipeline seja **executável por terceiros** — a banca ou um avaliador sem acesso ao banco local consegue reproduzir tudo |
| **Risco / limitação** | Duas fontes podem divergir. Mitigado pela auditoria de 37/37 colunas contra o UCI |
| **Onde** | `src/data_loader.py` → `carregar_dados()` |

### D-02 · Classificar as colunas em contínuas / binárias / nominais manualmente

| | |
|---|---|
| **Alternativa rejeitada** | Inferência automática de tipo por dtype ou cardinalidade |
| **Justificativa** | 29 das 36 colunas são `int64`, mas 9 delas são **códigos de categoria** (`course=9254` não é maior que `course=171`). Nenhuma inferência automática distinguiria `course` de `age` — ambos são inteiros. A classificação seguiu a documentação de Realinho et al. (2022) |
| **Risco / limitação** | Trabalho manual, sujeito a erro humano; precisa ser revisto se o dataset mudar |
| **Onde** | `src/data_loader.py` → `COLUNAS_CONTINUAS`, `COLUNAS_BINARIAS`, `COLUNAS_NOMINAIS`, `tipologia()` |

### D-03 · Tratar `application_order` como ordinal (numérica)

| | |
|---|---|
| **Alternativa rejeitada** | One-hot, junto com as outras nominais |
| **Justificativa** | Aqui a ordenação é **real e informativa**: quem entrou na 1ª opção escolheu o curso; quem entrou na 6ª aceitou o que sobrou. A distância entre 1 e 2 tem o mesmo sentido que entre 5 e 6 |
| **Risco / limitação** | Assume linearidade do efeito, que pode não ser exata |
| **Onde** | `src/data_loader.py` → `COLUNAS_CONTINUAS` |

### D-04 · Corrigir as notas corrompidas em vez de descartar as colunas

| | |
|---|---|
| **Alternativa rejeitada** | (a) Descartar as duas colunas; (b) usar o UCI diretamente e ignorar o problema do banco; (c) deixar como está |
| **Justificativa** | São as **duas variáveis mais preditivas do dataset** — descartá-las inviabilizaria o trabalho. A correção é determinística porque toda nota não nula está entre 9,80 e 18,875, logo tem exatamente 2 dígitos inteiros e a posição da vírgula é única. Auditada contra o UCI: **3.230 valores reconstruídos, zero divergências em 37/37 colunas** |
| **Risco / limitação** | A regra é específica desta base e deste tipo de corrupção. Se o banco for recarregado com outro formato, a validação de faixa aborta a execução com erro explícito |
| **Onde** | `src/data_loader.py` → `reparar_notas()`; auditoria em `01_analise_exploratoria.ipynb` §4.1; detalhes em [doc 02](02-qualidade-e-integridade.md) |

### D-05 · **Não** remover outliers

| | |
|---|---|
| **Alternativa rejeitada** | Remoção por IQR, winsorização, ou *clipping* nos percentis 1 e 99 |
| **Justificativa** | Evidência direta: registros com ao menos um outlier têm **45,7% de evasão**, contra **15,6%** dos demais — diferença de 30,1 pp. Os outliers **são o sinal**, não ruído. Removê-los eliminaria preferencialmente os alunos em risco, ou seja, a própria população-alvo |
| **Risco / limitação** | Modelos lineares são sensíveis a valores extremos; mitigado pela padronização e pela presença de modelos de árvore |
| **Onde** | `01_analise_exploratoria.ipynb` §10 |

### D-06 · **Não** imputar valores

| | |
|---|---|
| **Alternativa rejeitada** | Pipeline de imputação preventivo (mediana, KNN) |
| **Justificativa** | Não há valores ausentes — 0 nulos em 4.424 × 37. Imputação preventiva adicionaria complexidade sem função |
| **Risco / limitação** | O pipeline **não está preparado** para dados faltantes. Aplicação a dados reais brasileiros exigiria essa etapa. Registrado nas limitações |
| **Onde** | `01_analise_exploratoria.ipynb` §4 |

### D-07 · Manter as variáveis macroeconômicas

| | |
|---|---|
| **Alternativa rejeitada** | Remover `unemployment_rate`, `inflation_rate`, `gdp` |
| **Justificativa** | Removê-las sem evidência seria seleção arbitrária. Sua importância medida é baixa (< 1,5% cada), então o custo de mantê-las é mínimo |
| **Risco / limitação** | **Funcionam como identificador aproximado da coorte de ingresso.** O modelo pode usá-las para "descobrir o ano" — sinal real, mas que **não generaliza para anos futuros**. Limitação de generalização temporal registrada |
| **Onde** | `01_analise_exploratoria.ipynb` §8 |

### D-08 · Criar `is_dropout` apenas para análise descritiva

| | |
|---|---|
| **Alternativa rejeitada** | Usar a variável nas features (seria vazamento direto do alvo) |
| **Justificativa** | Necessária para calcular correlação ponto-bisserial com o alvo. Existe **somente** dentro das células de análise e nunca entra em `X` |
| **Risco / limitação** | Nenhum, desde que a separação seja mantida. Verificável: `metadados.json` lista as 105 features e `is_dropout` não está entre elas |
| **Onde** | `01_analise_exploratoria.ipynb` §8 |

---

## Fase 3 — Preparação dos Dados

### D-09 · Remover a classe `Enrolled` (formulação binária)

| | |
|---|---|
| **Alternativa rejeitada** | Classificação multiclasse com as 3 classes (**implementada e disponível**, ativável em uma linha) |
| **Justificativa** | (1) `Enrolled` não é desfecho, é **censura à direita** — o aluno ainda pode se formar ou evadir; (2) evidência empírica: sua mediana cai **entre** as outras duas em todas as variáveis, sem região própria; (3) é a formulação predominante na literatura desta base, o que preserva comparabilidade; (4) SHAP é muito mais legível com saída binária |
| **Risco / limitação** | **A limitação mais importante do trabalho.** O modelo é treinado em trajetórias concluídas, mas seria aplicado sobre alunos ativos — a população excluída. Diferença estrutural entre população de treino e de aplicação |
| **Onde** | `02_preprocessamento.ipynb` §2; constante `FORMULACAO` |

### D-10 · `Dropout` como classe positiva (código 1)

| | |
|---|---|
| **Alternativa rejeitada** | `Graduate` como positiva (ordem alfabética faria isso por padrão) |
| **Justificativa** | Custo assimétrico: falso negativo (evadido não detectado) perde a única chance de intervir; falso positivo custa uma conversa. Recall e precisão precisam se referir à classe de interesse |
| **Risco / limitação** | Nenhum — é convenção necessária para que as métricas sejam legíveis |
| **Onde** | `02_preprocessamento.ipynb` §2; `metadados.json` → `classe_positiva` |

### D-11 · Separar treino/teste **antes** de qualquer transformação

| | |
|---|---|
| **Alternativa rejeitada** | Transformar tudo e separar depois (mais simples de escrever) |
| **Justificativa** | `StandardScaler`, `OneHotEncoder` e SMOTE são **parâmetros estimados dos dados**. Ajustá-los no conjunto completo transmite informação do teste ao modelo, e a métrica final mede memorização. O erro é **silencioso** — produz números melhores, não piores |
| **Risco / limitação** | Nenhum. É a ordem correta |
| **Onde** | `02_preprocessamento.ipynb` §4 |

### D-12 · Split 80/20 com `stratify=y`

| | |
|---|---|
| **Alternativa rejeitada** | 70/30, ou split sem estratificação |
| **Justificativa** | 80/20 é o padrão da literatura e deixa 726 registros de teste — suficiente para estimativas estáveis. A estratificação preserva a taxa de evasão nos dois lados (0,3915 vs 0,3912) |
| **Risco / limitação** | **Partição única** — sem intervalo de confiança para as métricas de teste |
| **Onde** | `02_preprocessamento.ipynb` §4 |

### D-13 · `StandardScaler` nas contínuas, binárias intocadas

| | |
|---|---|
| **Alternativa rejeitada** | `MinMaxScaler`, `RobustScaler`, ou padronizar tudo |
| **Justificativa** | Escalas incompatíveis (notas 0–20, idade 17–70, PIB −4 a 4) exigem padronização para a Regressão Logística. As binárias já estão em 0/1 — padronizá-las apenas dificultaria a leitura dos coeficientes sem ganho. `RobustScaler` seria alternativa dado o volume de outliers, mas como os outliers **são sinal** (D-05), reduzir sua influência seria contraproducente |
| **Risco / limitação** | `StandardScaler` é sensível a outliers; aceito conscientemente |
| **Onde** | `02_preprocessamento.ipynb` §5 |

### D-14 · One-hot com `min_frequency=0.01`

| | |
|---|---|
| **Alternativa rejeitada** | (a) One-hot puro (200 dummies); (b) *target encoding*; (c) *label encoding* (manteria ordenação falsa); (d) agrupamento manual de categorias |
| **Justificativa** | 200 categorias em 9 variáveis, 131 delas com < 1% de presença. O agrupamento automático reduz para **78 dummies**, preserva as categorias com sustentação estatística e é **aprendido só no treino** (`handle_unknown="infrequent_if_exist"` torna a transformação robusta a categoria nova no teste). *Target encoding* foi rejeitado por risco de vazamento |
| **Risco / limitação** | O grupo `infrequent_sublevel` mistura categorias heterogêneas, dificultando a interpretação SHAP dessa dummy específica |
| **Onde** | `02_preprocessamento.ipynb` §5 |

### D-15 · SMOTE aplicado **apenas** ao treino

| | |
|---|---|
| **Alternativa rejeitada** | (a) SMOTE antes do split (**erro grave e comum**); (b) `class_weight='balanced'`; (c) *undersampling*; (d) nenhum balanceamento |
| **Justificativa** | Um ponto sintético é interpolado de alunos reais. Se ele for para o treino e seus "pais" para o teste, o modelo terá visto uma versão borrada dos registros de teste. `class_weight` seria alternativa legítima, mas o SMOTE é o procedimento predominante na literatura desta base |
| **Risco / limitação** | Interpola sobre dummies, gerando valores fracionários onde só existiriam 0/1. `SMOTENC` seria mais rigoroso — ver D-16 |
| **Onde** | `02_preprocessamento.ipynb` §6; verificado pela asserção "teste NÃO balanceado" |

### D-16 · SMOTE clássico em vez de `SMOTENC`

| | |
|---|---|
| **Alternativa rejeitada** | `SMOTENC`, que trata colunas categóricas de forma discreta |
| **Justificativa** | Comparabilidade com a literatura desta base, que usa SMOTE clássico. O efeito prático é pequeno porque as dummies representam parcela menor da variância |
| **Risco / limitação** | Valores fracionários em dummies nos 630 registros sintéticos. **Ressalva explicitamente registrada** no notebook e nesta documentação |
| **Onde** | `02_preprocessamento.ipynb` §6.4 |

### D-17 · **Nenhuma** seleção de variáveis

| | |
|---|---|
| **Alternativa rejeitada** | `SelectKBest`, RFE, ou corte por importância mínima (19 features têm < 0,1%) |
| **Justificativa** | (1) O objetivo é **interpretabilidade**, e uma variável irrelevante no agregado pode ser decisiva para um subgrupo — exatamente o que o SHAP revela; (2) sem restrição computacional (105 features, 2.904 registros, treino < 1 s); (3) seleção agressiva sem justificativa é decisão arbitrária questionável pela banca |
| **Risco / limitação** | Modelo mais complexo que o mínimo necessário; gráficos SHAP com mais variáveis para exibir |
| **Onde** | `02_preprocessamento.ipynb` §7.3 |

### D-18 · Salvar o treino em **duas** versões (com e sem SMOTE)

| | |
|---|---|
| **Alternativa rejeitada** | Salvar apenas o treino balanceado |
| **Justificativa** | A validação cruzada da Fase 4 **precisa** do treino natural, para que o SMOTE seja reaplicado por dobra (ver D-19). Salvar só o balanceado forçaria um erro metodológico na fase seguinte |
| **Risco / limitação** | Dois arquivos a mais; risco de usar o errado — mitigado por comentários explícitos e pela verificação de equivalência |
| **Onde** | `02_preprocessamento.ipynb` §8 |

---

## Fase 4 — Modelagem

### D-19 · SMOTE dentro do `imblearn.Pipeline` na validação cruzada

| | |
|---|---|
| **Alternativa rejeitada** | Rodar `cross_validate` / `GridSearchCV` direto sobre o treino já balanceado |
| **Justificativa** | Com o SMOTE aplicado ao conjunto inteiro, pontos sintéticos derivados de um aluno da dobra de **validação** aparecem nas dobras de **treino** — o modelo valida contra vizinhos do que já viu, e a métrica infla. O `imblearn.Pipeline` aplica reamostragem **só no `fit`**, nunca no `predict` |
| **Risco / limitação** | Nenhum. É a implementação correta. Verificado: o pipeline reproduz exatamente o SMOTE da Fase 3 |
| **Onde** | `03_treinamento_modelos.ipynb` §2 → `construir_pipeline()` |

### D-20 · `StratifiedKFold(5, shuffle=True)`

| | |
|---|---|
| **Alternativa rejeitada** | `KFold` simples, 10 dobras, ou validação cruzada aninhada |
| **Justificativa** | Estratificação é obrigatória com classes desbalanceadas. 5 dobras é o padrão e deixa ~580 registros por dobra de validação. `shuffle=True` evita dependência da ordem original das linhas |
| **Risco / limitação** | Validação cruzada **aninhada** seria mais rigorosa para estimar o desempenho pós-seleção de hiperparâmetros. Registrado como limitação |
| **Onde** | `03_treinamento_modelos.ipynb` §2.4 |

### D-21 · F1 da classe `Dropout` como métrica de refit

| | |
|---|---|
| **Alternativa rejeitada** | Acurácia (padrão do sklearn), ROC-AUC, ou recall puro |
| **Justificativa** | Acurácia é enganosa (piso trivial de 0,6088 com recall zero). **Recall puro é trivialmente maximizável** — "sempre `Dropout`" atinge recall 1,000 sinalizando todo mundo. F1 equilibra os dois e é a métrica de referência na literatura desta base |
| **Risco / limitação** | **F1 assume que os dois erros pesam igual** — o que este trabalho afirma ser falso, e a [Verificação 2](10-significancia.md) mostrou que a tensão é estatisticamente real: a Regressão Logística detecta significativamente mais evadidos no limiar 0,50 (p = 0,034). **A contradição foi resolvida, não apenas apontada** — ver [doc 04](04-modelagem.md), §9: com os dois limiares livres, o XGBoost detecta mais evadidos pelo mesmo custo em **31 de 31** pontos da faixa prática (recall ≤ 0,958), e a vantagem da Regressão Logística se restringe a uma banda estreita de recall muito alto (0,961–0,972) que exige convocar metade da turma. A magnitude em qualquer ponto pareado é de +1 a +5 evadidos |
| **Consequência** | A escolha do algoritmo é a **menos** importante de três decisões encadeadas: 1º a política de custo da instituição, 2º o limiar que a implementa, 3º o algoritmo. O trabalho vinha invertendo essa ordem |
| **Onde** | `03_treinamento_modelos.ipynb` §2.4 → `METRICA_ALVO` |

### D-22 · Regressão Logística mantida nos parâmetros padrão

| | |
|---|---|
| **Alternativa rejeitada** | Otimizar `C`, `penalty`, `solver` |
| **Justificativa** | É a **linha de base**. Uma linha de base otimizada deixa de cumprir sua função, que é estabelecer o patamar mínimo que a complexidade adicional precisa superar |
| **Risco / limitação** | A LR poderia ser melhor com ajuste — o que **fortaleceria** a conclusão de que o ganho dos modelos complexos é pequeno, não a enfraqueceria |
| **Onde** | `03_treinamento_modelos.ipynb` §4 |

### D-23 · `GridSearchCV` exaustivo em vez de `RandomizedSearchCV`

| | |
|---|---|
| **Alternativa rejeitada** | Busca aleatória com `n_iter` fixo, ou otimização bayesiana (Optuna) |
| **Justificativa** | 1.755 ajustes totais executam em ~2 minutos em 32 núcleos. Sendo viável, a busca completa é **mais simples de defender**: não há dúvida sobre "e se o melhor ponto não fosse sorteado?" |
| **Risco / limitação** | Não escalaria para grades maiores ou datasets maiores |
| **Onde** | `03_treinamento_modelos.ipynb` §5, §6 |

### D-24 · Desempate por PR-AUC

| | |
|---|---|
| **Alternativa rejeitada** | Aceitar a ordem do `sort` (não é critério); usar ROC-AUC; escolher o mais simples (Random Forest) |
| **Justificativa** | RF e XGBoost **empatam exatamente** em F1, recall, precisão e MCC (matriz 418/24/29/255). PR-AUC avalia o **ordenamento de probabilidades** em toda a faixa de limiares — mais informativa que ROC-AUC para classe positiva minoritária. A escolha resultante (XGBoost) é coerente com a CV, onde ele lidera em 5 de 6 métricas |
| **Risco / limitação** | Diferença de PR-AUC é pequena (0,9741 vs 0,9697) e provavelmente não significativa. O trabalho **não afirma** superioridade do XGBoost — trata os modelos como equivalentes na prática |
| **Onde** | `03_treinamento_modelos.ipynb` §8 |

### D-25 · Manter o limiar em 0,50 e reportar a análise separadamente

| | |
|---|---|
| **Alternativa rejeitada** | Adotar o limiar que maximiza F1 (0,64, F1 = 0,9191) como padrão |
| **Justificativa** | 0,50 preserva comparabilidade com a literatura. Otimizar o limiar **no conjunto de teste** e reportar a métrica resultante seria uma forma sutil de sobreajuste ao teste — a métrica deixaria de ser uma estimativa honesta de desempenho fora da amostra |
| **Risco / limitação** | Deixa-se 0,013 de F1 "na mesa". Compensado por não contaminar a estimativa de desempenho |
| **Onde** | `03_treinamento_modelos.ipynb` §9 |

### D-26 · Salvar o pipeline completo, não apenas o classificador

| | |
|---|---|
| **Alternativa rejeitada** | `joblib.dump(modelo)` apenas |
| **Justificativa** | O pipeline com SMOTE (inerte na predição) mais o `preprocessador.joblib` da Fase 3 permitem predizer para um aluno novo a partir dos dados brutos, sem reconstruir nenhuma etapa manualmente |
| **Risco / limitação** | Arquivo maior; acoplamento à versão das bibliotecas — mitigado pelo `requirements.txt` com versões fixas |
| **Onde** | `03_treinamento_modelos.ipynb` §10 |

### D-27 · Não realizar teste de significância entre modelos

| | |
|---|---|
| **Alternativa rejeitada** | Teste de McNemar, ou *bootstrap* dos intervalos de confiança |
| **Justificativa** | Decisão consciente de **escopo**. Em compensação, o texto **evita afirmar superioridade** de qualquer modelo e trata os três como equivalentes na prática — que é a conclusão que os dados sustentam sem o teste |
| **Risco / limitação** | Limitação declarada. É pergunta legítima da banca, e a resposta é: a conclusão do trabalho não depende de o XGBoost ser melhor. Se o teste apontasse equivalência estatística, **reforçaria** a tese central (o sinal está nos dados, não no algoritmo) |
| **Onde** | Registrado em [doc 07](07-limitacoes-e-trabalhos-futuros.md) |

---

## Fase 5 — Interpretabilidade

### D-32 · `TreeExplainer` em vez de `KernelExplainer`

| | |
|---|---|
| **Alternativa rejeitada** | `KernelExplainer` (agnóstico ao modelo) ou `PermutationExplainer` |
| **Justificativa** | O modelo final é baseado em árvores, e o TreeSHAP calcula os valores de Shapley **exatos** em tempo polinomial. `KernelExplainer` faria amostragem e devolveria aproximações, sem ganho algum aqui. A exatidão foi conferida: erro máximo de aditividade de **3,8 × 10⁻⁶** |
| **Risco / limitação** | Nenhum. É a escolha correta para o tipo de modelo |
| **Onde** | `04_interpretabilidade_shap.ipynb` §3 |

### D-33 · Explicar o **conjunto de teste**, não o treino

| | |
|---|---|
| **Alternativa rejeitada** | Calcular SHAP sobre o treino (mais dados: 3.534 contra 726) ou sobre a base completa |
| **Justificativa** | O treino contém **630 alunos sintéticos** do SMOTE. Explicar um aluno que não existe não tem sentido substantivo, e as médias globais ficariam contaminadas por pontos interpolados. O teste é composto exclusivamente de alunos reais, na proporção natural de evasão |
| **Risco / limitação** | Amostra menor para as estatísticas globais (726 alunos). Aceito: a alternativa comprometeria a interpretação |
| **Onde** | `04_interpretabilidade_shap.ipynb` §3 |

### D-34 · Somar as *dummies* para recompor as variáveis categóricas

| | |
|---|---|
| **Alternativa rejeitada** | Reportar apenas o ranking das 105 features codificadas |
| **Justificativa** | A **linearidade** dos valores de Shapley torna a soma matematicamente exata, não uma aproximação. Sem ela, `Curso` apareceria com 0,0965 (sua melhor dummy) em vez de 0,2332 — subestimada em **2,4×**. Responder "quais fatores importam?" exige o ranking recomposto; "quais features do modelo importam?" é outra pergunta |
| **Risco / limitação** | A soma dá magnitude, **não direção** — não existe "valor alto de curso". Para categóricas, a leitura útil é por categoria específica |
| **Onde** | `04_interpretabilidade_shap.ipynb` §7 |

### D-35 · Medir direção por Spearman entre valor e SHAP

| | |
|---|---|
| **Alternativa rejeitada** | Ler a direção visualmente no *beeswarm*; usar Pearson |
| **Justificativa** | A banca pode pedir o número, e leitura visual não é reportável. Spearman em vez de Pearson porque interessa a consistência da **ordem** do efeito, não sua linearidade — o modelo é de árvores e produz degraus, não retas |
| **Risco / limitação** | Um ρ próximo de zero com impacto alto é ambíguo entre "efeito não monotônico" e "efeito nulo". Resolvido examinando o *dependence plot* da variável |
| **Onde** | `04_interpretabilidade_shap.ipynb` §6 |

### D-36 · Incluir um **falso negativo** entre os casos individuais

| | |
|---|---|
| **Alternativa rejeitada** | Apresentar apenas casos corretamente classificados (o que a maioria dos trabalhos faz) |
| **Justificativa** | Um trabalho que só mostra acertos esconde a informação de que a instituição mais precisa: **em que tipo de aluno o sistema não pode ser confiado**. O caso C revelou o teto intrínseco do problema — aluno que evadiu tendo perfil protetor em todos os indicadores disponíveis |
| **Risco / limitação** | Expõe a limitação do modelo. É intencional: sustenta a recomendação R-6 (priorização, não veredito) |
| **Onde** | `04_interpretabilidade_shap.ipynb` §10 |

### D-37 · Converter os limiares para unidades reais

| | |
|---|---|
| **Alternativa rejeitada** | Reportar os pontos de virada em valores padronizados |
| **Justificativa** | "Valor padronizado abaixo de −0,175" não é acionável por uma secretaria; "até 3 aprovações no semestre" é. A inversão do `StandardScaler` transforma o achado estatístico em **gatilho operacional** |
| **Risco / limitação** | Os limiares valem para esta base; uma instituição brasileira precisaria recalculá-los |
| **Onde** | `04_interpretabilidade_shap.ipynb` §9 |

---

## Decisões transversais

### D-28 · `random_state=42` em toda operação estocástica

| | |
|---|---|
| **Onde aplicado** | `train_test_split`, `SMOTE`, `StratifiedKFold`, `LogisticRegression`, `RandomForestClassifier`, `XGBClassifier`, `mutual_info_classif`, geração de predições aleatórias |
| **Justificativa** | Reprodutibilidade exata — requisito de trabalho acadêmico |
| **Risco / limitação** | Os resultados são condicionais a **uma** semente. Repetir com sementes diferentes daria uma estimativa de variabilidade — não feito, registrado como limitação |

### D-29 · Credenciais fora do código

| | |
|---|---|
| **Alternativa rejeitada** | Credenciais no notebook (prática comum e insegura) |
| **Justificativa** | `.env` no `.gitignore`, lido via `python-dotenv`. `.env.example` documenta as chaves sem valores. Nome de tabela validado com regex antes de entrar na query; senha passa por `quote_plus` |
| **Onde** | `src/data_loader.py` → `_config_mysql()`, `carregar_do_mysql()` |

### D-30 · Verificações automáticas que abortam a execução

| | |
|---|---|
| **Justificativa** | Transformar afirmações metodológicas em **testes executáveis**. Uma afirmação em markdown pode ficar desatualizada quando o código muda; uma asserção não |
| **Verificações ativas** | Faixa 0–20 após correção das notas (Fase 2); 7 asserções de integridade dos artefatos, incluindo "teste NÃO balanceado" (Fase 3); equivalência do SMOTE no pipeline e reprodutibilidade das predições após recarga do modelo (Fase 4) |
| **Onde** | `src/data_loader.py`; `02_preprocessamento.ipynb` §8; `03_treinamento_modelos.ipynb` §2, §10 |

### D-31 · Rótulos em português, inclusive para dummies

| | |
|---|---|
| **Justificativa** | Os gráficos vão direto para a monografia. `rotulo()` traduz nomes canônicos e também dummies geradas pelo one-hot: `application_mode_1` → "Modalidade de ingresso = 1"; `course_infrequent_sublevel` → "Curso = outras (raras)" |
| **Onde** | `src/data_loader.py` → `ROTULOS_PT`, `rotulo()` |

---

## Verificações finais

### D-38 · Manter o modelo não calibrado, com ressalva de linguagem

| | |
|---|---|
| **Alternativa rejeitada** | Adotar `CalibratedClassifierCV` com isotonic e substituir o modelo final |
| **Justificativa** | Ganho de Brier de apenas 0,0004; a miscalibração se concentra em faixa pouco populada, onde nenhum calibrador tem dados suficientes; o ROC-AUC é **idêntico**, logo a ordenação — de que todos os resultados dependem — não muda |
| **Risco / limitação** | Afirmações probabilísticas diretas ficam imprecisas; resolvido chamando a saída de "escore de risco". Superestimação sistemática de 3,7 pp registrada |
| **Onde** | `06_calibracao.ipynb` §4–5; [doc 09](09-calibracao.md) |

### D-39 · Reportar o resultado da semente 42 com a expectativa entre sementes ao lado

| | |
|---|---|
| **Alternativa rejeitada** | (a) Trocar para uma semente "média"; (b) reportar a média das 10 sementes como resultado principal |
| **Justificativa** | Trocar a semente após ver os resultados seria seleção oportunista — precisamente o que se quer evitar. A semente 42 foi fixada *a priori* ([D-28](#d-28--random_state42-em-toda-operação-estocástica)) e seu resultado é correto para aquela partição. Reportar as duas informações lado a lado é honesto e mais informativo que qualquer uma isolada |
| **Risco / limitação** | O número de destaque (F1 = 0,9059) é otimista; mitigado pela menção explícita da expectativa de ≈ 0,88 |
| **Onde** | `07_significancia.ipynb` §4; [doc 10](10-significancia.md) |

### D-40 · Manter o modelo, com monitoramento de equidade como condição de uso

| | |
|---|---|
| **Alternativa rejeitada** | (a) Declarar o modelo "auditado e justo"; (b) remover gênero e variáveis socioeconômicas do modelo |
| **Justificativa** | A auditoria não encontrou disparidade significativa, o que não justifica intervenção. Remover atributos sensíveis é ineficaz — o modelo os reconstrói por correlação — e reduziria a capacidade de detectar justamente os alunos socioeconomicamente vulneráveis, que são a população de interesse. Mas o poder estatístico limitado impede declarar equidade |
| **Risco / limitação** | Disparidades de 10 a 20 pp em subgrupos pequenos permaneceriam indetectadas. Mitigado pela exigência de monitoramento contínuo (R-5 revisada) |
| **Onde** | `08_auditoria_equidade.ipynb` §4; [doc 11](11-equidade.md) |

---

## Sumário: as cinco decisões que mais afetam os resultados

Se a banca tiver tempo para apenas cinco perguntas, provavelmente serão estas:

| # | Decisão | Impacto |
|---|---|---|
| **D-09** | Remover `Enrolled` | Define o problema. Cria a diferença entre população de treino e de aplicação — a limitação central do trabalho |
| **D-04** | Corrigir as notas corrompidas | Sem isso, as duas variáveis mais preditivas estariam arruinadas e todos os resultados seriam inválidos |
| **D-15 / D-19** | SMOTE só no treino, e por dobra na CV | Sem isso, todas as métricas seriam otimistas de forma indetectável |
| **D-21** | F1 como métrica de decisão | Determina qual modelo vence e assume simetria de custos que o trabalho nega |
| **D-05** | Manter outliers | Preserva 54,7% dos registros, incluindo a maior parte da população em risco |

> **Nota acrescentada após a Fase 5.** A decisão **D-17** (não fazer seleção de variáveis) deixou de ser
> apenas uma escolha de princípio e passou a ter **justificativa empírica**: a taxa de desemprego tem
> correlação marginal de −0,004 com a evasão — qualquer filtro univariado a descartaria — e revelou-se
> a **9ª variável mais importante** do modelo. Ver [doc 08](08-interpretabilidade-shap.md), §5.
