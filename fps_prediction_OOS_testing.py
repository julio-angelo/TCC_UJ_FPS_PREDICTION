import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import resample
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

# --- 1. DEFINIR OS ARQUIVOS ---
arquivo_treino = 'df1_processado.csv'  # O arquivo principal para ensinar o modelo
arquivo_teste  = '360_test.csv'  # O arquivo de teste para validar (nunca visto antes)
#arquivo_teste  = 'test_new_clean.csv'  # O arquivo de teste para validar (nunca visto antes)

print(f"Carregando dados de TREINO: {arquivo_treino}...")
print(f"Carregando dados de TESTE:  {arquivo_teste}...")

try:
    df_train = pd.read_csv(arquivo_treino)
    df_test = pd.read_csv(arquivo_teste)
    
    # Limpeza básica em ambos
    df_train.dropna(inplace=True)
    df_test.dropna(inplace=True)
    
    print(f"Dados carregados. Treino: {len(df_train)} linhas | Teste: {len(df_test)} linhas.")

except FileNotFoundError as e:
    print(f"ERRO: Arquivo não encontrado: {e}")
    exit()

# --- 2. PREPARAR OS DADOS (GARANTINDO A CONSISTÊNCIA) ---

# Separar Alvo (y) e Features (X) do TREINO
y_train = df_train['FrameTime']
X_train = df_train.drop('FrameTime', axis=1)

# Separar Alvo (y) e Features (X) do TESTE
# IMPORTANTE: O arquivo de teste DEVE ter a coluna FrameTime para podermos calcular o erro depois.
if 'FrameTime' not in df_test.columns:
    print("ERRO: O arquivo de teste precisa ter a coluna 'FrameTime' para avaliarmos a precisão.")
    exit()

y_test = df_test['FrameTime']

# TRUQUE DE SEGURANÇA: 
# Garantimos que o X_test tenha EXATAMENTE as mesmas colunas e na mesma ordem que o X_train.
# Se o arquivo de teste tiver colunas a mais, elas serão ignoradas.
# Se tiver colunas a menos, isso vai gerar um erro avisando você.
try:
    X_test = df_test[X_train.columns]
except KeyError as e:
    print(f"\nERRO CRÍTICO: O arquivo de teste não tem todas as colunas que o treino usou.")
    print(f"Coluna faltando: {e}")
    print("Certifique-se de que ambos os CSVs passaram pelo mesmo processo de limpeza.")
    exit()

feature_names = X_train.columns.tolist()

# --- 3. TREINAMENTO (Apenas no df_train) ---
model = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=42)
print("\nIniciando o treinamento do modelo (usando APENAS o arquivo de treino)...")
start_time = time.time()

model.fit(X_train, y_train)

duration = time.time() - start_time
print(f"Treinamento concluído em: {duration:.4f} segundos.")

# --- 4. PREVISÃO E AVALIAÇÃO (Apenas no df_test) ---
print("\nFazendo previsões no arquivo de teste...")
predictions = model.predict(X_test)

# Métricas
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
mape = mean_absolute_percentage_error(y_test, predictions) * 100
r2 = r2_score(y_test, predictions)

print("\n--- QUALIDADE DO MODELO NO NOVO DATASET ---")
print(f"Erro Médio Absoluto (MAE): {mae:.4f} ms")
print(f"Raiz do Erro Quadrático Médio (RMSE): {rmse:.4f} ms")
print(f"Erro Percentual Absoluto Médio (MAPE): {mape:.4f}%")
print(f"Coeficiente de Determinação (R²): {r2:.4f}")
print("-------------------------------------------")

# --- 6. ANÁLISE DE INCERTEZA (BOOTSTRAP) ---
print("\nIniciando Bootstrap para Intervalos de Confiança (95%)...")

n_iterations = 1000
stats_mae = []
stats_rmse = []
stats_r2 = []
stats_mape = []

# Converter para array numpy para facilitar a manipulação
y_test_arr = np.array(y_test)
pred_arr = np.array(predictions)

for i in range(n_iterations):
    # Reamostragem com substituição (mantendo a correspondência entre Real e Predito)
    # O random_state=i garante reprodutibilidade se necessário, ou remova para aleatoriedade total
    y_sample, pred_sample = resample(y_test_arr, pred_arr, replace=True, n_samples=len(y_test_arr))
    
    # Recalcular métricas para esta amostra
    b_mae = mean_absolute_error(y_sample, pred_sample)
    b_rmse = np.sqrt(mean_squared_error(y_sample, pred_sample))
    b_r2 = r2_score(y_sample, pred_sample)
    b_mape = mean_absolute_percentage_error(y_sample, pred_sample) * 100
    
    stats_mae.append(b_mae)
    stats_rmse.append(b_rmse)
    stats_r2.append(b_r2)
    stats_mape.append(b_mape)

# Função auxiliar para calcular os limites inferior (2.5%) e superior (97.5%)
def get_intervals(data):
    lower = np.percentile(data, 2.5)
    upper = np.percentile(data, 97.5)
    mean_val = np.mean(data)
    return mean_val, lower, upper

# Calcular e exibir
mae_m, mae_l, mae_u = get_intervals(stats_mae)
rmse_m, rmse_l, rmse_u = get_intervals(stats_rmse)
r2_m, r2_l, r2_u = get_intervals(stats_r2)
mape_m, mape_l, mape_u = get_intervals(stats_mape)

print("\n--- RESULTADOS COM INTERVALO DE CONFIANÇA (95%) ---")
print(f"MAE:  {mae_m:.4f} ms  [{mae_l:.4f}, {mae_u:.4f}]")
print(f"RMSE: {rmse_m:.4f} ms  [{rmse_l:.4f}, {rmse_u:.4f}]")
print(f"MAPE: {mape_m:.4f} %   [{mape_l:.4f}, {mape_u:.4f}]")
print(f"R²:   {r2_m:.4f}      [{r2_l:.4f}, {r2_u:.4f}]")
print("---------------------------------------------------")

# --- 6. Exibir as Features Mais Importantes ---
print("\n--- TOP 20 FEATURES MAIS IMPORTANTES ---")
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print(feature_importance_df.head(20).to_string(index=False))
print("---------------------------------------")

# --- 5. VISUALIZAÇÃO ---
# Gráfico 1: Previsão vs Real
plt.figure(figsize=(10, 10))
plt.scatter(y_test, predictions, alpha=0.5, label='Amostras de Teste')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Previsão Perfeita')
plt.title(f'Validação Externa: Treino({arquivo_treino}) vs Teste({arquivo_teste})')
plt.xlabel('FrameTime Real (ms)')
plt.ylabel('FrameTime Previsto (ms)')
plt.legend()
plt.grid(True)

# Gráfico 2: Comparação de Distribuição (para ver se os arquivos são muito diferentes)
plt.figure(figsize=(12, 6))
plt.hist(y_train, bins=50, alpha=0.5, label=f'Treino: {arquivo_treino}', density=True, color='blue')
plt.hist(y_test, bins=50, alpha=0.5, label=f'Teste: {arquivo_teste}', density=True, color='green')
plt.title('Comparação das Distribuições de FrameTime')
plt.xlabel('FrameTime (ms)')
plt.ylabel('Frequência')
plt.legend()
plt.grid(True, alpha=0.3)



# --- . ANÁLISE DE RESÍDUOS ---

# 1. Calcular os resíduos (Diferença entre Real e Previsto)
residuos = y_test - predictions

# 2. Calcular a média dos resíduos (Deve ser próxima de zero)
media_residuos = np.mean(residuos)
print(f"\n--- ANÁLISE DE RESÍDUOS ---")
print(f"Média dos Resíduos: {media_residuos:.4f} (Ideal: próximo de 0)")

# 3. Plotar a Distribuição (Histograma + Curva KDE)
plt.figure(figsize=(10, 6))
sns.histplot(residuos, kde=True, color='blue', bins=30)
plt.axvline(x=0, color='red', linestyle='--', label='Zero (Erro Nulo)')
plt.axvline(x=media_residuos, color='green', linestyle='-', label=f'Média ({media_residuos:.2f})')

plt.title('Distribuição dos Resíduos (Erros de Previsão)')
plt.xlabel('Erro (ms) -> (Negativo = Previu a mais, Positivo = Previu a menos)')
plt.ylabel('Frequência')
plt.legend()
plt.grid(True, alpha=0.3)

# 4. (Opcional) Teste de Normalidade (Q-Q Plot)
# Se os pontos seguirem a linha vermelha, a distribuição é normal
plt.figure(figsize=(8, 6))
stats.probplot(residuos, dist="norm", plot=plt)
plt.title('Q-Q Plot dos Resíduos')
plt.grid(True)





# --- VISUALIZAÇÃO DE FEATURE IMPORTANCE ---

# 1. Extrair as importâncias e os nomes das colunas
importances = model.feature_importances_
feature_names = X_train.columns

# 2. Criar um DataFrame para organizar os dados
feature_df = pd.DataFrame({
    'Feature': feature_names, 
    'Importance': importances
})

# 3. Ordenar de forma decrescente (do mais importante para o menos)
feature_df = feature_df.sort_values(by='Importance', ascending=False)

# 4. Configurar o Gráfico
# Vamos pegar apenas as TOP 20 para o gráfico não ficar poluído
top_n = 20
top_features = feature_df.head(top_n)

plt.figure(figsize=(12, 10)) # Tamanho grande para caber os nomes
plt.barh(top_features['Feature'], top_features['Importance'], color='#2c3e50')

# Inverter o eixo Y para que a feature #1 fique no topo
plt.gca().invert_yaxis()

# 5. Estilização
plt.xlabel('Grau de Importância Relativa (0 a 1)', fontsize=12)
plt.title(f'Top {top_n} Features Determinantes para o FrameTime', fontsize=14, pad=20)
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Ajustar layout para não cortar os nomes das features
plt.tight_layout()

plt.show()
