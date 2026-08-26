# 02 — Qualidade e Integridade dos Dados

> **Escopo:** falha de integridade encontrada na base MySQL, o raciocínio da correção e a auditoria
> que a validou.
> **Implementação:** `src/data_loader.py` — funções `_slug()`, `_nota_esta_integra()`,
> `_reparar_nota()`, `reparar_notas()`
> **Auditoria reexecutável:** `01_analise_exploratoria.ipynb`, seção 4.1

Este documento trata do achado metodológico mais relevante da Fase 2. Ele está separado porque não é
um detalhe de engenharia: é um caso concreto de por que **as verificações padrão de qualidade de dados
são insuficientes**, e o raciocínio tem valor próprio para a monografia.

---

## 1. Contexto

Os dados foram carregados do banco MySQL local (`tcc`, MySQL 8.4.8), organizado com tabelas
dimensionais (`dim_course`, `dim_occupation`, `dim_qualification`, `dim_marital_status`,
`dim_nationality`, `dim_application_mode`) e a tabela achatada `data` com 4.424 linhas × 37 colunas.

A inspeção do schema — e não a inspeção dos valores — revelou o problema.

---

## 2. As duas falhas encontradas

### 2.1 Notas semestrais corrompidas por separador de milhar

`SHOW COLUMNS FROM data` mostrou duas colunas com o tipo errado:

```
Curricular units 1st sem (grade)    text     ← deveria ser DOUBLE
Curricular units 2nd sem (grade)    text     ← deveria ser DOUBLE
```

As demais colunas de nota (`Previous qualification (grade)`, `Admission grade`) estavam corretamente
como `double`. Ao inspecionar os valores das duas colunas `text`:

```sql
SELECT DISTINCT `Curricular units 1st sem (grade)` FROM data LIMIT 15;
```

```
'0.0'
'14.0'
'13.428.571.428.571.400'     ←  ?!
'12.333.333.333.333.300'     ←  ?!
'13.3'
'13.875'
'1.330.625'                  ←  ?!
'114.375'                    ←  ?!  (descoberto depois — ver §4)
```

**Mecanismo da corrupção.** Trata-se do erro clássico de importação de CSV em planilha com
configuração regional diferente da do arquivo: o **ponto decimal foi reinterpretado como separador de
milhar**, o número foi convertido em inteiro e reformatado com pontos a cada três casas antes de ser
gravado no banco.

| Valor real | Como ficou gravado |
|---|---|
| `13,42857142857140` | `"13.428.571.428.571.400"` |
| `11,4375` | `"114.375"` |
| `13,30625` | `"1.330.625"` |

### 2.2 BOM no nome da primeira coluna

A primeira coluna vinha do MySQL como `ï»¿Marital status` — os três bytes do BOM UTF-8 (`EF BB BF`)
interpretados como latin-1 durante a importação do CSV. Após a normalização Unicode padrão, o nome
resultava em `imarital_status` (o `ï` decompõe em `i` + trema, e o trema é descartado no ASCII).

---

## 3. Por que essas falhas são graves — e invisíveis

### O impacto

Lidas com um `float()` ingênuo, as notas corrompidas produziriam valores como **114,375 e 13.428.571
em uma escala de 0 a 20**. As consequências em cascata:

| Etapa afetada | Efeito |
|---|---|
| `describe()` | Média e desvio-padrão sem sentido |
| Matriz de correlação | Correlações distorcidas — a variável mais preditiva do dataset, arruinada |
| `StandardScaler` | Média e escala calculadas sobre valores absurdos, comprimindo todos os valores válidos |
| Modelos | Treinados sobre a variável de maior importância totalmente deformada |
| SHAP | Explicações construídas sobre valores irreais |

### Por que passou por todas as verificações

Este é o ponto central:

| Verificação padrão | Resultado | Detectou? |
|---|---|---|
| `df.isnull().sum()` | 0 nulos | ❌ |
| `df.duplicated().sum()` | 0 duplicatas | ❌ |
| Colunas constantes | nenhuma | ❌ |
| `df.describe()` | executou normalmente | ❌ |
| **Tipo declarado no schema × tipo esperado** | `TEXT` onde deveria ser `DOUBLE` | ✅ |
| **Domínio da variável (0–20)** | valores de 114 e 13 milhões | ✅ |

> **Ausência de nulos não é sinônimo de qualidade.** As três verificações que todo tutorial de EDA
> ensina passaram sem alarme. O que encontrou o erro foi confrontar o **tipo declarado** com o tipo
> esperado, e a **faixa observada** com o domínio conhecido da variável.

---

## 4. A correção — e por que ela é determinística

Reverter a corrupção exige responder: *removidos os pontos, onde vai a vírgula decimal?*

Para `"13428571428571400"`, os candidatos plausíveis são `1,3428...` e `13,428...`. **Ambos caem na
escala válida de 0 a 20.** A ambiguidade é real e, à primeira vista, irreversível.

### O argumento que resolve a ambiguidade

A resposta veio do **domínio da variável**, verificado na fonte oficial:

```
Notas semestrais não nulas no dataset original:
   mínimo  =  9,80
   máximo  = 18,875
   média   = 12,72
   nº de notas não nulas < 10 :  1  (de 7.260)
```

**Toda nota diferente de zero tem exatamente dois dígitos na parte inteira.** E as notas zero foram
gravadas como `"0.0"` — íntegras, com um único ponto, portanto fora do escopo da correção.

Isso torna a posição da vírgula **única**: após o 2º dígito. Não há chute.

### A regra implementada

```python
_DIGITOS_PARTE_INTEIRA = 2

def _nota_esta_integra(texto):
    """Íntegra = no máximo um ponto, no máximo dois dígitos inteiros, valor em [0, 20]."""
    if texto.count(".") > 1:
        return False
    if len(texto.split(".")[0]) > _DIGITOS_PARTE_INTEIRA:
        return False
    return 0.0 <= float(texto) <= 20.0

def _reparar_nota(bruto):
    texto = str(bruto).strip().replace(",", ".")
    digitos = texto.replace(".", "")
    if _nota_esta_integra(texto):
        return float(texto)
    return float(f"{digitos[:2]}.{digitos[2:]}")
```

### O erro intermediário que vale registrar

A **primeira versão** da regra classificava como corrompido apenas o valor com **mais de um ponto**.
Ela falhou: valores como `"114.375"` têm um único ponto, mas três dígitos na parte inteira — o real é
`11,4375`.

O erro foi capturado por uma **validação de faixa** colocada após a correção:

```python
fora = df[col].dropna()
fora = fora[(fora < 0) | (fora > 20)]
if len(fora):
    raise ValueError(f"{col}: {len(fora)} nota(s) fora da escala 0–20 após a correção...")
```

```
ValueError: curricular_units_1st_sem_grade: 39 nota(s) fora da escala 0–20
após a correção (ex.: [114.375, 132.375, 115.875]). Verifique a carga no banco.
```

> **Lição metodológica:** a validação de domínio não serve apenas para encontrar o problema original —
> ela protege contra uma **correção incompleta**. Sem essa checagem, 39 valores errados teriam
> atravessado silenciosamente todo o pipeline. Uma correção de dados sem validação posterior é uma
> hipótese não testada.

A validação permanece ativa no código: se o banco for recarregado com um formato diferente, a execução
**para com erro explícito** em vez de produzir resultados silenciosamente errados.

### Correção do BOM

```python
texto = texto.lstrip("﻿").removeprefix("ï»¿").lstrip("﻿")
```

Trata as duas formas: o BOM UTF-8 legítimo e a versão *mojibake* que o MySQL devolve quando o arquivo
foi lido como latin-1.

---

## 5. Auditoria: validação contra a fonte oficial

A regra não foi aceita por ser plausível — foi **conferida contra o dataset publicado** (UCI id=697),
linha a linha.

### Metodologia

1. Carregar a tabela `data` do MySQL e aplicar a correção
2. Baixar o dataset oficial do UCI
3. Normalizar os nomes de coluna dos dois lados para o mesmo padrão
4. Comparar **todas as 37 colunas**, na ordem original das linhas

### Resultado

```
Comparação linha a linha com o UCI (todas as 37 colunas):
   [OK] 37/37 colunas idênticas.
   A base local, após a correção, reproduz exatamente o dataset publicado.
```

| Item | Resultado |
|---|---|
| Colunas comparadas | 37 de 37 |
| Colunas divergentes | **0** |
| Valores de nota reconstruídos | **3.230** (1.672 no 1º sem. + 1.558 no 2º) |
| Percentual afetado | 37,8% e 35,2% das linhas, respectivamente |
| Ordem das linhas | idêntica ao original |
| Divergências residuais | **0 em 4.424 linhas** |

A auditoria está implementada como célula reexecutável na **seção 4.1 do Notebook 1**, e degrada com
elegância: se não houver internet, informa que não pôde executar e segue — a correção não depende dela.

> **Valor para a defesa:** esta auditoria é uma **evidência de procedência dos dados**. Diante da
> pergunta "como você garante que os dados que usou são os dados publicados?", a resposta é uma célula
> que pode ser executada na hora.

---

## 6. Recomendação para o banco de dados

**A correção acontece em memória, durante a carga.** As colunas continuam como `TEXT` e com os valores
deformados no MySQL. Qualquer consulta feita fora deste pipeline — um relatório, um dashboard, outro
script — lerá os valores errados.

Recomenda-se corrigir a origem:

```sql
-- 1. Backup antes de qualquer alteração
CREATE TABLE data_backup AS SELECT * FROM data;

-- 2. Corrigir os valores (mesma regra validada: remove pontos, vírgula após o 2º dígito)
UPDATE data
SET `Curricular units 1st sem (grade)` = CAST(
      CONCAT(
        SUBSTRING(REPLACE(`Curricular units 1st sem (grade)`, '.', ''), 1, 2), '.',
        SUBSTRING(REPLACE(`Curricular units 1st sem (grade)`, '.', ''), 3)
      ) AS DECIMAL(18,14))
WHERE CHAR_LENGTH(`Curricular units 1st sem (grade)`)
      - CHAR_LENGTH(REPLACE(`Curricular units 1st sem (grade)`, '.', '')) > 1
   OR LOCATE('.', `Curricular units 1st sem (grade)`) > 3;

-- (repetir para `Curricular units 2nd sem (grade)`)

-- 3. Converter o tipo
ALTER TABLE data
  MODIFY `Curricular units 1st sem (grade)` DOUBLE,
  MODIFY `Curricular units 2nd sem (grade)` DOUBLE;

-- 4. Validar: nenhuma nota deve ficar fora da escala
SELECT COUNT(*) FROM data
WHERE `Curricular units 1st sem (grade)` NOT BETWEEN 0 AND 20
   OR `Curricular units 2nd sem (grade)` NOT BETWEEN 0 AND 20;
-- resultado esperado: 0

-- 5. Corrigir o BOM no nome da primeira coluna
ALTER TABLE data CHANGE `ï»¿Marital status` `Marital status` INT;
```

> ⚠️ **Este script não foi executado.** Ele altera dados de forma destrutiva e a decisão é do autor do
> trabalho. Recomenda-se rodar o passo 1 (backup) e o passo 4 (validação) obrigatoriamente.
> O pipeline **não depende** dessa correção — `reparar_notas()` é idempotente e continuará funcionando
> mesmo depois que o banco for consertado (colunas já numéricas são ignoradas).

---

## 7. Robustez adicional implementada no loader

Além da correção, o `data_loader.py` foi endurecido para tolerar diferenças entre o banco local e o
dataset de referência:

| Mecanismo | O que resolve |
|---|---|
| `_slug()` | Normaliza qualquer grafia de nome de coluna para `snake_case` ASCII, removendo BOM, acentos e apóstrofos (`Mother's qualification` → `mothers_qualification`) |
| `ALIASES` | Mapeia nomes alternativos que uma view poderia usar (`situacao`, `status`, `resultado` → `target`) e descarta chaves substitutas (`id_aluno`, `sk_aluno`) |
| `reparar_notas()` idempotente | Ignora colunas que já venham numéricas — funciona igual com MySQL corrigido ou com o fallback UCI |
| Validação de faixa | Aborta com erro explícito se a correção produzir valor fora de [0, 20] |
| Validação do nome da tabela | `re.fullmatch(r"[A-Za-z0-9_$.]+", tabela)` antes de montar a query |
| Fallback automático | Qualquer exceção na conexão MySQL → carga do UCI, com o motivo informado |

---

## 8. Síntese

### O que aconteceu

Duas falhas de integridade na base MySQL, invisíveis às verificações padrão de qualidade:
**3.230 valores de nota deformados** por separador de milhar indevido (≈36% de cada coluna afetada) e
um **BOM** corrompendo o nome da primeira coluna.

### Como foi resolvido

Correção determinística — justificada pelo domínio da variável, não por conveniência — implementada no
loader, protegida por validação de faixa e **auditada contra a fonte oficial com zero divergências em
37 de 37 colunas**.

### As três lições para a monografia

1. **Ausência de nulos não é qualidade.** `isnull()`, `duplicated()` e `describe()` passaram sem
   alarme. O erro apareceu ao confrontar o tipo declarado no schema com o tipo esperado, e a faixa
   observada com o domínio conhecido da variável. **Validar tipo e domínio é parte do entendimento dos
   dados**, não um detalhe de engenharia.

2. **Uma correção de dados precisa ser justificável, não conveniente.** Esta só é defensável porque o
   domínio da variável torna a posição do separador decimal única, e porque foi verificada contra a
   fonte original. Registrar esse raciocínio é o que distingue **tratamento de dado** de
   **manipulação de dado** — e é exatamente o tipo de distinção que uma banca sonda.

3. **Toda correção precisa de validação posterior.** A primeira versão da regra estava incompleta e
   deixaria 39 valores errados passarem. Foi a validação de faixa — e não a inspeção visual — que
   encontrou a falha. Uma correção sem verificação é uma hipótese não testada.
