# Predição de Evasão no Ensino Superior

**TCC — Desenvolvimento de modelo preditivo interpretável de machine learning para identificação de
alunos em risco de evasão no ensino superior**
Curso de Sistemas de Informação — UEMG

Pipeline completo de machine learning seguindo a metodologia **CRISP-DM**, com foco em
**interpretabilidade**: o modelo não apenas identifica quem está em risco, mas explica **por quê** — o
que permite escolher a intervenção adequada a cada caso.

---

## Resultado principal

Modelo selecionado: **XGBoost**, avaliado em 726 alunos reais nunca vistos pelo modelo.

| Métrica (classe `Dropout`) | Valor |
|---|---:|
| **F1-score** | **0,9059** |
| Recall | 0,8979 |
| Precisão | 0,9140 |
| ROC-AUC | 0,9747 |
| PR-AUC | 0,9741 |
| Acurácia | 0,9270 |

> De cada 100 alunos que evadem, o modelo identifica **90 nesta partição dos dados**.
> De cada 100 alunos sinalizados, **91** realmente evadiriam.
>
> **Ressalva de partição:** a [Verificação 2](docs/10-significancia.md) mostrou que a semente 42 produziu
> uma partição favorável. O valor típico esperado em outras partições é de **84** detectados de cada 100
> evadidos (faixa 82–86) e F1 de **≈ 0,88**. A precisão não apresentou o mesmo viés.

### O achado central

O ganho do XGBoost sobre a linha de base (Regressão Logística) é de apenas **+1,4% em F1**. Os três
modelos testados convergem para o mesmo patamar porque **o sinal preditivo está nos dados — desempenho
acadêmico e situação financeira — e não na sofisticação do algoritmo**.

Isso é favorável ao objetivo do trabalho: como não há custo relevante de desempenho em escolher um
modelo mais explicável, a **interpretabilidade passa a ser critério legítimo de decisão**, não concessão.

---

## Dataset

| | |
|---|---|
| **Nome** | Predict Students' Dropout and Academic Success |
| **Fonte** | UCI Machine Learning Repository, id **697** (Realinho et al., 2022) |
| **Contexto** | Instituto Politécnico de Portalegre, Portugal — ingressos de 2008 a 2019 |
| **Registros** | 4.424 estudantes |
| **Atributos** | 36 variáveis independentes + variável-alvo |
| **Fonte primária usada** | Banco MySQL local (`tcc.data`), com fallback automático para o UCI |

**Variável-alvo:** `Graduate` (49,9%) · `Dropout` (32,1%) · `Enrolled` (17,9%)

---

## Estrutura do repositório

```
tcc-dropout-prediction/
├── 01_analise_exploratoria.ipynb      Fase 2 — Entendimento dos dados       (48 células)
├── 02_preprocessamento.ipynb          Fase 3 — Preparação dos dados         (27 células)
├── 03_treinamento_modelos.ipynb       Fase 4 — Modelagem                    (50 células)
├── 04_interpretabilidade_shap.ipynb   Fase 5 — Interpretabilidade (SHAP)    (49 células)
├── 05_relatorio_final.ipynb           Fase 6 — Relatório consolidado        (23 células)
├── 06_calibracao.ipynb                Verificação — calibração de probabilidade (15 células)
├── 07_significancia.ipynb             Verificação — bootstrap, McNemar, sementes (26 células)
├── 08_auditoria_equidade.ipynb        Verificação — equidade algorítmica     (14 células)
│
├── src/
│   └── data_loader.py                 Carga MySQL/UCI, padronização de colunas,
│                                      correção de integridade, rótulos em português
│
├── docs/                              Documentação técnica detalhada — ver índice abaixo
│
├── dados/
│   ├── dataset_limpo.csv              4.424 × 37 — saída do NB1
│   ├── X_train.csv / y_train.csv      3.534 × 105 — treino APÓS SMOTE
│   ├── X_train_sem_smote.csv          2.904 × 105 — treino natural (base da validação cruzada)
│   ├── X_test.csv / y_test.csv        726 × 105 — teste intocado
│   ├── metadados.json                 Formulação, classes, lista das 105 features
│   ├── metricas_modelos.csv           Métricas dos 3 modelos (teste + validação cruzada)
│   ├── analise_limiar.csv             Curva precisão/recall/F1 por limiar de decisão
│   ├── shap_*.csv                     5 tabelas de interpretabilidade
│   ├── shap_valores_teste.npz         Valores SHAP completos (726 × 105)
│   └── shap_casos_individuais.json    Os três casos explicados em detalhe
│
├── modelos/
│   ├── preprocessador.joblib          ColumnTransformer ajustado no treino
│   ├── melhor_modelo.joblib           XGBoost — pipeline completo
│   ├── modelo_*.joblib                Os três modelos, para reanálise
│   └── info_melhor_modelo.json        Hiperparâmetros, métricas, limiar ótimo
│
├── figuras/                           29 PNGs em 200 dpi, prontos para a monografia
├── .env.example                       Template de credenciais (o .env real não é versionado)
└── requirements.txt
```

---

## Documentação

> **[📄 RELATÓRIO CONSOLIDADO](RELATORIO-CONSOLIDADO.md)** — o registro único e autocontido de tudo o
> que foi construído e de todas as conclusões, incluindo as três verificações finais. Comece por aqui.

A documentação detalhada por fase está em [`docs/`](docs/) e registra **todo o raciocínio, os achados e
os resultados** de cada etapa.

| Documento | Conteúdo |
|---|---|
| [00 — Visão geral](docs/00-visao-geral.md) | Objetivo, dataset, arquitetura do pipeline, ambiente, como reproduzir |
| [01 — Entendimento dos dados](docs/01-entendimento-dos-dados.md) | Análise exploratória, distribuições, correlações, outliers |
| [02 — Qualidade e integridade](docs/02-qualidade-e-integridade.md) | Falha de integridade encontrada no MySQL e sua correção auditada |
| [03 — Preparação dos dados](docs/03-preparacao-dos-dados.md) | Tratamento de `Enrolled`, codificação, split, SMOTE |
| [04 — Modelagem](docs/04-modelagem.md) | Três modelos, busca de hiperparâmetros, comparação, seleção |
| [05 — Decisões metodológicas](docs/05-decisoes-metodologicas.md) | 40 decisões, cada uma com alternativa rejeitada e risco assumido |
| [06 — Resultados e figuras](docs/06-resultados-e-figuras.md) | 31 tabelas e 29 figuras prontas para transcrição na monografia |
| [07 — Limitações e trabalhos futuros](docs/07-limitacoes-e-trabalhos-futuros.md) | 15 limitações e 9 propostas de continuidade, priorizadas |
| [08 — Interpretabilidade com SHAP](docs/08-interpretabilidade-shap.md) | Importância, direção, limiares operacionais, casos individuais, 7 recomendações |
| [09 — Calibração de probabilidade](docs/09-calibracao.md) | O escore é probabilidade real? Efeito sobre a análise de limiar |
| [10 — Significância estatística](docs/10-significancia.md) | *Bootstrap*, McNemar em dois escopos, 10 sementes |
| [11 — Equidade algorítmica](docs/11-equidade.md) | Recall e FPR por subgrupo, *equalized odds*, poder estatístico |
| [12 — Revisão final](docs/12-revisao-final.md) | Changelog das correções de coerência interna aplicadas após revisão crítica |

O **Notebook 5** consolida tudo em formato de relatório, e inclui uma demonstração de uso ponta a ponta:
`avaliar_aluno()` recebe os dados brutos de um aluno e devolve probabilidade, explicação individual e
encaminhamento sugerido.

---

## Como reproduzir

```bash
# 1. Ambiente virtual
python -m venv .venv
.venv\Scripts\activate                 # Windows
pip install -r requirements.txt

# 2. Credenciais do banco (OPCIONAL)
#    Sem .env, o pipeline baixa os dados do UCI automaticamente.
copy .env.example .env
#    preencher MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE, MYSQL_TABLE

# 3. Executar os notebooks na ordem
jupyter lab
```

**Tempo total:** ~5 minutos. O gargalo é a busca de hiperparâmetros do Notebook 3 (1.755 ajustes de
modelo nas duas buscas, 1.760 com a validação cruzada da linha de base), paralelizada em `n_jobs=-1`.

**Reprodutibilidade:** `random_state=42` em toda operação estocástica. Executar na ordem produz
resultados idênticos aos documentados.

**Segurança:** as credenciais nunca aparecem no código — são lidas de variáveis de ambiente via
`python-dotenv`. O `.env` está no `.gitignore`.

---

## Rigor metodológico

Os pontos abaixo são os que mais afetam a validade dos resultados, e cada um tem **verificação
automática** no código:

| Prática | Verificação |
|---|---|
| Separação treino/teste **antes** de qualquer transformação | Desvio residual do teste após padronização (média −0,022, não 0) |
| `StandardScaler` e `OneHotEncoder` ajustados **só no treino** | idem |
| SMOTE aplicado **só no treino** | Asserção "teste NÃO balanceado" aborta a execução se violada |
| SMOTE reaplicado **por dobra** na validação cruzada | `imblearn.Pipeline`; equivalência verificada contra o SMOTE do NB2 |
| Nenhuma métrica calculada sobre dados sintéticos | Teste mantido em 726 registros, 39,1% de evasão natural |
| Notas corrompidas na origem corrigidas e auditadas | 37/37 colunas idênticas à fonte oficial, 0 divergências |
| Modelo salvo reproduz as predições | Asserção após recarga do `.joblib` |
| Explicação SHAP reconstrói a predição | Aditividade conferida: erro máximo 3,8 × 10⁻⁶ |
| Reconstrução da partição é fiel | Semente 42 reproduz exatamente os valores publicados (NB7, NB8) |
| Split reprodutível a partir do dataset limpo | Reconstruído no NB5 e comparado a `X_test.csv` |
| Todos os entregáveis presentes e coerentes | 8 conferências finais no NB5 abortam se algo divergir |

Um erro em qualquer desses pontos produziria métricas **melhores**, não piores — e por isso passaria
despercebido sem verificação explícita.

---

## Situação do projeto

| Etapa | Notebook | Situação |
|---|---|---|
| Fase 2 — Entendimento dos dados | `01_analise_exploratoria.ipynb` | ✅ concluída |
| Fase 3 — Preparação dos dados | `02_preprocessamento.ipynb` | ✅ concluída |
| Fase 4 — Modelagem | `03_treinamento_modelos.ipynb` | ✅ concluída |
| Fase 5 — Interpretabilidade (SHAP) | `04_interpretabilidade_shap.ipynb` | ✅ concluída |
| Fase 6 — Relatório final | `05_relatorio_final.ipynb` | ✅ concluída |
| Verificação 1 — Calibração | `06_calibracao.ipynb` | ✅ concluída |
| Verificação 2 — Significância estatística | `07_significancia.ipynb` | ✅ concluída |
| Verificação 3 — Equidade algorítmica | `08_auditoria_equidade.ipynb` | ✅ concluída |

**Pipeline completo.** As seis fases do CRISP-DM estão executadas e documentadas, e as três
verificações posteriores testaram a robustez dos resultados — a mais consequente delas, a de
significância, revelou que a partição da semente 42 é otimista e obrigou a revisão de todos os números
operacionais do projeto. O changelog dessas correções está em
[`docs/12-revisao-final.md`](docs/12-revisao-final.md).

---

## Referências

- Realinho, V.; Machado, J.; Baptista, L.; Martins, M. V. (2022). *Predicting Student Dropout and
  Academic Success*. Data, 7(11), 146.
- Chawla, N. V.; Bowyer, K. W.; Hall, L. O.; Kegelmeyer, W. P. (2002). *SMOTE: Synthetic Minority
  Over-sampling Technique*. Journal of Artificial Intelligence Research, 16, 321–357.
- Chen, T.; Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. KDD '16.
- Lundberg, S. M.; Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS.
- Wirth, R.; Hipp, J. (2000). *CRISP-DM: Towards a Standard Process Model for Data Mining*.
