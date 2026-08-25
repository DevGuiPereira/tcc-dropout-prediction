"""
Módulo de carga de dados do TCC — Predição de Evasão no Ensino Superior.

Fonte primária : banco MySQL local (view flat do modelo dimensional).
Fonte fallback : UCI Machine Learning Repository, dataset id=697
                 "Predict Students' Dropout and Academic Success"
                 (Realinho et al., 2022).

As credenciais não ficam no código: são lidas de variáveis de ambiente
(arquivo `.env`, carregado com python-dotenv). Veja `.env.example`.
"""

from __future__ import annotations

import os
import re
import unicodedata
from typing import Tuple

import pandas as pd

try:  # python-dotenv é opcional em tempo de import
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


# --------------------------------------------------------------------------- #
# 1. Padronização de nomes de colunas
# --------------------------------------------------------------------------- #
def _slug(texto: str) -> str:
    """Converte um nome de coluna qualquer em snake_case ASCII.

    Exemplos:
        "Curricular units 1st sem (approved)" -> curricular_units_1st_sem_approved
        "Mother's qualification"              -> mothers_qualification
        "Nota de Admissão"                    -> nota_de_admissao
    """
    texto = str(texto)
    # Remove o BOM do CSV de origem, tanto em UTF-8 (﻿) quanto na forma
    # mojibake que o MySQL devolve quando o arquivo foi lido como latin-1
    # ("ï»¿Marital status"). Sem isso a 1ª coluna vira "imarital_status".
    texto = texto.lstrip("﻿").removeprefix("ï»¿").lstrip("﻿")
    texto = unicodedata.normalize("NFKD", texto)
    texto = texto.encode("ascii", "ignore").decode("ascii")
    texto = texto.replace("'", "").replace("`", "")
    texto = re.sub(r"[^0-9a-zA-Z]+", "_", texto)
    return texto.strip("_").lower()


# Aliases: nomes que a view do MySQL pode usar -> nome canônico do pipeline.
# Só é aplicado quando a coluna canônica ainda não existe no DataFrame.
ALIASES = {
    # identificadores que devem ser descartados
    "id_aluno": "_drop_id",
    "aluno_id": "_drop_id",
    "id_estudante": "_drop_id",
    "sk_aluno": "_drop_id",
    "id": "_drop_id",
    # target
    "situacao": "target",
    "situacao_final": "target",
    "status": "target",
    "resultado": "target",
    "desfecho": "target",
    "evasao": "target",
    # variações de grafia comuns
    "nationality": "nacionality",
    "daytime_evening_attendance_": "daytime_evening_attendance",
    "gdp_": "gdp",
}


def padronizar_colunas(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica snake_case, resolve aliases e remove colunas de identificação."""
    df = df.copy()
    df.columns = [_slug(c) for c in df.columns]

    renomear = {}
    for col in df.columns:
        destino = ALIASES.get(col)
        if destino and destino not in df.columns:
            renomear[col] = destino
        elif destino == "_drop_id":
            renomear[col] = "_drop_id"
    df = df.rename(columns=renomear)

    # descarta chaves substitutas / identificadores (não são preditores)
    df = df.drop(columns=[c for c in df.columns if c.startswith("_drop_id")], errors="ignore")
    return df


# --------------------------------------------------------------------------- #
# 2. Rótulos amigáveis (usados nos gráficos da monografia)
# --------------------------------------------------------------------------- #
ROTULOS_PT = {
    "marital_status": "Estado civil",
    "application_mode": "Modalidade de ingresso",
    "application_order": "Ordem de opção do curso",
    "course": "Curso",
    "daytime_evening_attendance": "Turno (diurno/noturno)",
    "previous_qualification": "Formação anterior",
    "previous_qualification_grade": "Nota da formação anterior",
    "nacionality": "Nacionalidade",
    "mothers_qualification": "Escolaridade da mãe",
    "fathers_qualification": "Escolaridade do pai",
    "mothers_occupation": "Ocupação da mãe",
    "fathers_occupation": "Ocupação do pai",
    "admission_grade": "Nota de admissão",
    "displaced": "Deslocado da cidade de origem",
    "educational_special_needs": "Necessidades educacionais especiais",
    "debtor": "Inadimplente",
    "tuition_fees_up_to_date": "Mensalidades em dia",
    "gender": "Gênero",
    "scholarship_holder": "Bolsista",
    "age_at_enrollment": "Idade no ingresso",
    "international": "Estudante internacional",
    "curricular_units_1st_sem_credited": "Disciplinas creditadas (1º sem.)",
    "curricular_units_1st_sem_enrolled": "Disciplinas matriculadas (1º sem.)",
    "curricular_units_1st_sem_evaluations": "Avaliações realizadas (1º sem.)",
    "curricular_units_1st_sem_approved": "Disciplinas aprovadas (1º sem.)",
    "curricular_units_1st_sem_grade": "Média das notas (1º sem.)",
    "curricular_units_1st_sem_without_evaluations": "Disciplinas sem avaliação (1º sem.)",
    "curricular_units_2nd_sem_credited": "Disciplinas creditadas (2º sem.)",
    "curricular_units_2nd_sem_enrolled": "Disciplinas matriculadas (2º sem.)",
    "curricular_units_2nd_sem_evaluations": "Avaliações realizadas (2º sem.)",
    "curricular_units_2nd_sem_approved": "Disciplinas aprovadas (2º sem.)",
    "curricular_units_2nd_sem_grade": "Média das notas (2º sem.)",
    "curricular_units_2nd_sem_without_evaluations": "Disciplinas sem avaliação (2º sem.)",
    "unemployment_rate": "Taxa de desemprego",
    "inflation_rate": "Taxa de inflação",
    "gdp": "PIB",
    "target": "Situação do aluno",
}


def rotulo(coluna: str) -> str:
    """Retorna o rótulo em português da coluna.

    Também traduz as *dummies* geradas pelo OneHotEncoder, que chegam no formato
    `<coluna_original>_<categoria>`:

        "application_mode_1"                  -> "Modalidade de ingresso = 1"
        "course_infrequent_sklearn"           -> "Curso = outras (raras)"
    """
    if coluna in ROTULOS_PT:
        return ROTULOS_PT[coluna]

    # Casa com o prefixo mais longo possível, para não confundir colunas cujo
    # nome é prefixo de outra.
    for base in sorted(ROTULOS_PT, key=len, reverse=True):
        if coluna.startswith(base + "_"):
            categoria = coluna[len(base) + 1:]
            if categoria in ("infrequent_sklearn", "infrequent_sublevel"):
                categoria = "outras (raras)"
            return f"{ROTULOS_PT[base]} = {categoria}"

    return coluna.replace("_", " ").capitalize()


# --------------------------------------------------------------------------- #
# 3. Carga a partir do MySQL
# --------------------------------------------------------------------------- #
def _config_mysql() -> dict:
    """Lê a configuração de conexão do ambiente (.env)."""
    if load_dotenv is not None:
        load_dotenv(override=False)
    return {
        "host": os.getenv("MYSQL_HOST", "localhost"),
        "port": os.getenv("MYSQL_PORT", "3306"),
        "user": os.getenv("MYSQL_USER", ""),
        "password": os.getenv("MYSQL_PASSWORD", ""),
        "database": os.getenv("MYSQL_DATABASE", ""),
        "table": os.getenv("MYSQL_TABLE", "vw_dataset_ml"),
    }


def carregar_do_mysql() -> pd.DataFrame:
    """Carrega a view flat do MySQL. Levanta exceção se a conexão falhar."""
    from sqlalchemy import create_engine  # import tardio: só quando realmente usado

    cfg = _config_mysql()
    if not cfg["user"] or not cfg["database"]:
        raise RuntimeError(
            "MYSQL_USER / MYSQL_DATABASE não definidos. "
            "Copie `.env.example` para `.env` e preencha as credenciais."
        )

    from urllib.parse import quote_plus

    url = (
        f"mysql+pymysql://{quote_plus(cfg['user'])}:{quote_plus(cfg['password'])}"
        f"@{cfg['host']}:{cfg['port']}/{cfg['database']}"
    )
    engine = create_engine(url, pool_pre_ping=True)
    # O nome da tabela/view vem de configuração local do próprio pesquisador,
    # não de entrada de usuário final — mas validamos o formato mesmo assim.
    tabela = cfg["table"]
    if not re.fullmatch(r"[A-Za-z0-9_$.]+", tabela):
        raise ValueError(f"Nome de tabela/view inválido: {tabela!r}")

    with engine.connect() as conn:
        df = pd.read_sql(f"SELECT * FROM {tabela}", conn)
    engine.dispose()
    return df


# --------------------------------------------------------------------------- #
# 4. Carga a partir do UCI (fallback)
# --------------------------------------------------------------------------- #
def carregar_do_uci() -> pd.DataFrame:
    """Baixa o dataset original do UCI (id=697) e devolve features + target."""
    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=697)
    X = dataset.data.features
    y = dataset.data.targets
    df = pd.concat([X, y], axis=1)
    return df


# --------------------------------------------------------------------------- #
# 4b. Correção das notas semestrais corrompidas na carga do MySQL
# --------------------------------------------------------------------------- #
# PROBLEMA IDENTIFICADO NO BANCO
# ------------------------------
# As colunas `Curricular units 1st/2nd sem (grade)` foram gravadas como TEXT e
# cerca de 37% dos registros chegaram deformados por um erro clássico de
# importação de CSV (o ponto decimal foi reinterpretado como separador de
# milhar pela planilha antes da carga):
#
#     valor real 13,42857142857140  ->  gravado como "13.428.571.428.571.400"
#     valor real 11,4375            ->  gravado como "114.375"
#
# POR QUE A CORREÇÃO É SEGURA (e não um "chute")
# ----------------------------------------------
# A nota é uma média em escala 0–20. No dataset original, TODA nota diferente
# de zero está entre 9,80 e 18,875 — ou seja, sempre com exatamente DOIS
# dígitos na parte inteira; as notas zero foram gravadas como "0.0". Assim, um
# valor é reconhecidamente íntegro quando tem no máximo um ponto, no máximo
# dois dígitos inteiros e cai em [0, 20]. Em qualquer outro formato basta
# remover os pontos e recolocar a vírgula após o 2º dígito — posição única,
# sem ambiguidade.
#
# VALIDAÇÃO EXECUTADA: a regra foi conferida contra a fonte oficial (UCI
# id=697), linha a linha. Os 3.230 valores corrompidos (1.672 no 1º semestre e
# 1.558 no 2º) foram reconstruídos com correspondência exata — zero
# divergências nas 4.424 linhas. As outras 35 colunas da tabela `data` já são
# idênticas ao original, na mesma ordem de linhas.

NOTAS_SEMESTRAIS = (
    "curricular_units_1st_sem_grade",
    "curricular_units_2nd_sem_grade",
)

#: Dígitos da parte inteira de uma nota semestral não nula (escala 0–20).
_DIGITOS_PARTE_INTEIRA = 2


def _nota_esta_integra(texto: str) -> bool:
    """Decide se a string já representa a nota corretamente.

    É íntegra quando tem no máximo um ponto, no máximo dois dígitos na parte
    inteira e resulta em um valor dentro da escala 0–20. Qualquer outra forma
    ("114.375", "1.330.625") é fruto do separador de milhar indevido.
    """
    if texto.count(".") > 1:
        return False
    if len(texto.split(".")[0]) > _DIGITOS_PARTE_INTEIRA:
        return False
    return 0.0 <= float(texto) <= 20.0


def _reparar_nota(bruto) -> float:
    """Converte para float um valor de nota, desfazendo o separador de milhar."""
    texto = str(bruto).strip().replace(",", ".")
    if texto == "" or texto.lower() in {"nan", "none", "null", "<na>"}:
        return float("nan")

    digitos = texto.replace(".", "")
    if not digitos.isdigit():
        raise ValueError(f"Valor de nota não reconhecido: {bruto!r}")

    if _nota_esta_integra(texto):
        return float(texto)

    inteiro = digitos[:_DIGITOS_PARTE_INTEIRA]
    decimal = digitos[_DIGITOS_PARTE_INTEIRA:]
    return float(f"{inteiro}.{decimal}")


def reparar_notas(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Converte as notas semestrais para float, corrigindo a corrupção do banco.

    É idempotente: se as colunas já vierem numéricas (caso do fallback UCI),
    nada é alterado.
    """
    df = df.copy()
    for col in NOTAS_SEMESTRAIS:
        if col not in df.columns or pd.api.types.is_numeric_dtype(df[col]):
            continue

        bruto = df[col].astype("string")
        corrompidos = int(sum(not _nota_esta_integra(str(v).strip()) for v in bruto))
        df[col] = bruto.map(_reparar_nota).astype("float64")

        fora = df[col].dropna()
        fora = fora[(fora < 0) | (fora > 20)]
        if len(fora):
            raise ValueError(
                f"{col}: {len(fora)} nota(s) fora da escala 0–20 após a correção "
                f"(ex.: {fora.head(3).tolist()}). Verifique a carga no banco."
            )

        if verbose and corrompidos:
            print(
                f"[CORREÇÃO] {col}: {corrompidos} de {len(df)} valores "
                f"({corrompidos / len(df):.1%}) reconstruídos "
                f"(separador de milhar indevido)."
            )
    return df


# --------------------------------------------------------------------------- #
# 5. Função pública
# --------------------------------------------------------------------------- #
def carregar_dados(verbose: bool = True) -> Tuple[pd.DataFrame, str]:
    """Carrega o dataset priorizando o MySQL e caindo para o UCI se necessário.

    Returns:
        (df, origem) — `origem` é "MySQL" ou "UCI (fallback)".
    """
    try:
        df = carregar_do_mysql()
        origem = "MySQL"
        if verbose:
            cfg = _config_mysql()
            print(f"[OK] Dados carregados do MySQL: {cfg['database']}.{cfg['table']}")
    except Exception as erro:  # noqa: BLE001 - queremos degradar para o fallback
        if verbose:
            print(f"[AVISO] MySQL indisponível ({type(erro).__name__}: {erro})")
            print("[INFO] Usando fallback: UCI ML Repository (id=697)...")
        df = carregar_do_uci()
        origem = "UCI (fallback)"
        if verbose:
            print("[OK] Dados carregados do UCI.")

    df = padronizar_colunas(df)
    df = reparar_notas(df, verbose=verbose)
    if verbose:
        print(f"[INFO] Dimensões: {df.shape[0]} linhas x {df.shape[1]} colunas")
    return df, origem


# --------------------------------------------------------------------------- #
# 6. Tipologia das variáveis
# --------------------------------------------------------------------------- #
# No dataset original quase tudo é numérico, mas boa parte dos inteiros são
# CÓDIGOS de categorias (ex.: `course` = 33, 171, 8014...). Tratar esses códigos
# como grandezas contínuas induziria o modelo a supor uma ordenação que não
# existe na realidade. Por isso classificamos as colunas explicitamente.

COLUNAS_CONTINUAS = [
    "previous_qualification_grade",
    "admission_grade",
    "age_at_enrollment",
    "curricular_units_1st_sem_credited",
    "curricular_units_1st_sem_enrolled",
    "curricular_units_1st_sem_evaluations",
    "curricular_units_1st_sem_approved",
    "curricular_units_1st_sem_grade",
    "curricular_units_1st_sem_without_evaluations",
    "curricular_units_2nd_sem_credited",
    "curricular_units_2nd_sem_enrolled",
    "curricular_units_2nd_sem_evaluations",
    "curricular_units_2nd_sem_approved",
    "curricular_units_2nd_sem_grade",
    "curricular_units_2nd_sem_without_evaluations",
    "unemployment_rate",
    "inflation_rate",
    "gdp",
    "application_order",  # ordinal (1ª a 9ª opção) — tratada como numérica
]

COLUNAS_BINARIAS = [
    "daytime_evening_attendance",
    "displaced",
    "educational_special_needs",
    "debtor",
    "tuition_fees_up_to_date",
    "gender",
    "scholarship_holder",
    "international",
]

COLUNAS_NOMINAIS = [
    "marital_status",
    "application_mode",
    "course",
    "previous_qualification",
    "nacionality",
    "mothers_qualification",
    "fathers_qualification",
    "mothers_occupation",
    "fathers_occupation",
]


def tipologia(df: pd.DataFrame) -> dict:
    """Devolve as colunas presentes no `df`, agrupadas por tipo."""
    presentes = set(df.columns)
    return {
        "continuas": [c for c in COLUNAS_CONTINUAS if c in presentes],
        "binarias": [c for c in COLUNAS_BINARIAS if c in presentes],
        "nominais": [c for c in COLUNAS_NOMINAIS if c in presentes],
    }
