import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

# --- 1. CARREGAMENTO DOS ARQUIVOS ---
# Substitua pelos seus nomes reais
arquivo_treino = 'df1_processado.csv'
arquivo_teste  = '360_test.csv'

print(f"Carregando Treino: {arquivo_treino}...")
print(f"Carregando Teste:  {arquivo_teste}...")

try:
    df1 = pd.read_csv(arquivo_treino)
    df2 = pd.read_csv(arquivo_teste)
    df1.dropna(inplace=True)
    df2.dropna(inplace=True)
except FileNotFoundError:
    print("ERRO: Arquivos não encontrados.")
    exit()

# --- 2. ALINHAMENTO DE COLUNAS (CRUCIAL) ---
# Garante que usamos apenas as colunas que existem NOS DOIS arquivos
cols_treino = set(df1.columns)
cols_teste = set(df2.columns)
cols_comuns = list(cols_treino.intersection(cols_teste))

if 'FrameTime' in cols_comuns:
    cols_comuns.remove('FrameTime')

print(f"\nFeatures em comum encontradas: {len(cols_comuns)}")

# --- FUNÇÃO DE TREINO E AVALIAÇÃO ---
def avaliar_cenario(X_tr, y_tr, X_te, y_te, nome_cenario):
    print(f"\n>>> RODANDO: {nome_cenario}")
    
    model = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42)
    start = time.time()
    model.fit(X_tr, y_tr)
    print(f"Treino concluído em {time.time() - start:.2f}s")
    
    preds = model.predict(X_te)
    
    # Métricas
    mae = mean_absolute_error(y_te, preds)
    rmse = np.sqrt(mean_squared_error(y_te, preds))
    mape = mean_absolute_percentage_error(y_te, preds) * 100
    r2 = r2_score(y_te, preds)
    
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R2': r2, 'preds': preds, 'y_true': y_te}

# ==============================================================================
# CENÁRIO A: MIXED / SHUFFLE (Otimista)
# Juntamos os dois arquivos e misturamos tudo. O modelo vê pedaços de ambos.
# ==============================================================================
df_total = pd.concat([df1, df2], ignore_index=True)
X_total = df_total[cols_comuns]
y_total = df_total['FrameTime']

# Divide 80/20 misturado
X_train_A, X_test_A, y_train_A, y_test_A = train_test_split(X_total, y_total, test_size=0.2, shuffle=True, random_state=42)

res_A = avaliar_cenario(X_train_A, y_train_A, X_test_A, y_test_A, "Cenário A: Validação Interna (Shuffle)")

# ==============================================================================
# CENÁRIO B: OUT-OF-SAMPLE REAL (Rigoroso)
# Treina APENAS no Arquivo 1. Testa APENAS no Arquivo 2.
# ==============================================================================
X_train_B = df1[cols_comuns]
y_train_B = df1['FrameTime']
X_test_B  = df2[cols_comuns]
y_test_B  = df2['FrameTime']

res_B = avaliar_cenario(X_train_B, y_train_B, X_test_B, y_test_B, "Cenário B: Validação Externa (Out-of-Sample)")

# --- 3. RELATÓRIO COMPARATIVO ---
print(f"\n{'='*70}")
print("COMPARAÇÃO DE ESTRATÉGIAS DE VALIDAÇÃO")
print(f"{'='*70}")
print(f"{'Métrica':<10} | {'A: Shuffle (Interna)':<22} | {'B: OOS (Externa)':<22} | {'Variação'}")
print("-" * 70)
print(f"{'R²':<10} | {res_A['R2']:.4f}                 | {res_B['R2']:.4f}                 | {res_B['R2'] - res_A['R2']:.4f}")
print(f"{'MAE':<10} | {res_A['MAE']:.4f} ms              | {res_B['MAE']:.4f} ms              | {res_B['MAE'] - res_A['MAE']:.4f}")
print(f"{'MAPE':<10} | {res_A['MAPE']:.2f}%                | {res_B['MAPE']:.2f}%                | {res_B['MAPE'] - res_A['MAPE']:.2f}%")
print("-" * 70)

# --- 4. GRÁFICOS ---
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Gráfico A
axes[0].scatter(res_A['y_true'], res_A['preds'], alpha=0.2, color='blue')
axes[0].plot([0, 100], [0, 100], 'r--', lw=2) # Linha ideal
axes[0].set_title(f"Cenário A: Shuffle (Misturado)\nR² = {res_A['R2']:.3f}")
axes[0].set_xlabel("Real (ms)")
axes[0].set_ylabel("Previsto (ms)")
axes[0].grid(True, alpha=0.3)

# Gráfico B
axes[1].scatter(res_B['y_true'], res_B['preds'], alpha=0.2, color='orange')
axes[1].plot([0, 100], [0, 100], 'r--', lw=2) # Linha ideal
axes[1].set_title(f"Cenário B: Out-of-Sample (Separado)\nR² = {res_B['R2']:.3f}")
axes[1].set_xlabel("Real (ms)")
axes[1].set_ylabel("Previsto (ms)")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Gráfico de Resíduos
plt.figure(figsize=(10, 6))
sns.kdeplot(res_A['y_true'] - res_A['preds'], fill=True, label='A: Shuffle', color='blue', alpha=0.3)
sns.kdeplot(res_B['y_true'] - res_B['preds'], fill=True, label='B: OOS Real', color='orange', alpha=0.3)
plt.axvline(0, color='red', linestyle='--')
plt.title("Densidade dos Erros (Resíduos)")
plt.xlabel("Erro (ms)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
