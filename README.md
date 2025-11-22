# TCC_UJ_FPS_PREDICTION
O objetivo geral deste projeto é utilizar técnicas de Machine Learning para desenvolver e validar um modelo preditivo de métricas de desempenho para aplicações no Unreal Engine e, a partir dele, possibilitar maior flexibilidade para os desenvolvedores na tarefa de otimização.


# Unreal Engine Performance Prediction with Machine Learning

Este repositório contém o código-fonte e os datasets utilizados na pesquisa de conclusão de curso sobre **Predição de Desempenho Gráfico (FPS/FrameTime) em Unreal Engine utilizando Machine Learning**.

O projeto implementa um pipeline completo de Ciência de Dados:
1.  **Pré-processamento:** Limpeza de dados de telemetria bruta do *Unreal CSV Profiler*.
2.  **Treinamento e Validação (OOS):** Treinamento de modelo *Random Forest* e validação rigorosa em dados nunca vistos (*Out-of-Sample*).
3.  **Análise Comparativa:** Comparação científica entre métodos de validação (Shuffle Split vs. Block Split) para provar a robustez contra *overfitting*.

---

## 📋 Arquivos do Projeto


*    **df1_processado.csv:** Dataset de TREINO (Cenas variadas, limpo).
*    **360_test.csv:** Dataset de TESTE (Cena inédita, limpo).
*    **filtrar_colunas.py:** Script da Etapa 1 (Limpeza e Seleção de Features para saida do unreal csvprofiler).
*    **fps_prediction_OOS_testing.py:** Script da Etapa 2 (Treino, Teste Out-of-Sample e Feature Importance).
*    **model_performace.py:** Script da Etapa 3 (Comparação Metodológica entre Shuffle vs OOS).
*    **README.md:** Documentação do projeto.
*    **requirements.txt:** Lista de dependências do Python.

---

## 🚀 Como Executar o Pipeline

### 1. Pré-requisitos

Certifique-se de ter o Python (3.8+) instalado. Instale as dependências necessárias executando:

```bash
pip install -r requirements.txt
```

*(Veja o conteúdo do `requirements.txt` ao final deste documento)*

---

### 2. O Pipeline Passo a Passo

#### 🔹 Etapa 1: Limpeza de Dados (`filtrar_colunas.py`)
Este script processa o arquivo bruto gerado pelo comando `csvprofile` do Unreal Engine. Ele remove colunas vazias, corta as últimas linhas (geralmente instáveis na captura) e filtra apenas as colunas de telemetria relevantes para o estudo.

*   **Entrada:** Arquivo CSV bruto (ex: `Profile_Raw.csv`).
*   **Saída:** Arquivo CSV limpo pronto para ML.
*   **Como usar:**
    1.  Abra o script.
    2.  Edite a variável `input_csv_path` com o nome do seu arquivo bruto.
    3.  Execute:
    ```bash
    python filtrar_colunas.py
    ```
Para melhor entendimento sobre o Unreal CSV profiler visite: https://motiongorilla.com/articles/8/

#### 🔹 Etapa 2: Treinamento e Validação Real (`fps_prediction_OOS_testing.py`)
Este é o script principal de validação. Ele treina o modelo no dataset principal (`df1_processado.csv`) e testa sua capacidade de generalização em um arquivo separado (`360_test.csv`).

*   **O que ele faz:**
    *   Calcula métricas de erro (MAE, RMSE, MAPE, R²).
    *   Gera intervalos de confiança via *Bootstrap* (95%).
    *   Exibe o gráfico de **Feature Importance** (O que mais impacta o FPS?).
    *   Gera gráficos de Resíduos e Q-Q Plot.
*   **Como usar:**
    ```bash
    python fps_prediction_OOS_testing.py
    ```

#### 🔹 Etapa 3: Análise Comparativa de Performance (`model_performace.py`)
Este script serve para validação científica. Ele compara dois cenários para provar que o modelo é robusto:
1.  **Cenário A (Shuffle):** Mistura treino e teste (Validação Interna).
2.  **Cenário B (Out-of-Sample):** Mantém os arquivos separados (Validação Externa).

*   **Objetivo:** Demonstrar a diferença entre "decorar dados" e "aprender padrões".
*   **Como usar:**
    ```bash
    python model_performace.py
    ```

---

## 📊 Resultados Principais

Os experimentos demonstraram que o modelo é capaz de prever o tempo de quadro (*FrameTime*) com alta precisão em cenários inéditos.

| Métrica (Validação Externa) | Valor Obtido |
| :--- | :--- |
| **Erro Médio Absoluto (MAE)** | ~0.25 ms |
| **Erro Percentual (MAPE)** | ~1.45% |
| **Coeficiente R²** | ~0.75 |

> **Nota:** A análise de *Feature Importance* revelou que o gargalo predominante nos cenários testados foi **CPU-Bound** (`CPUUsage_Process`), seguido por latência de comandos de desenho (`DrawSceneCommand`).

---

## 🛠 Tecnologias Utilizadas

*   **Python 3.10+**
*   **Pandas:** Manipulação de dados.
*   **Scikit-Learn:** Algoritmo *Random Forest* e métricas.
*   **Matplotlib / Seaborn:** Visualização de dados e gráficos estatísticos.
*   **SciPy:** Análises estatísticas (Q-Q Plot).
