import pickle
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import optuna
import pandas as pd
import yfinance as yf
from sklearn.covariance import LedoitWolf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


# 1. COLETA DE DADOS E CÁLCULO DOS RETORNOS

tickers = [
    'B3SA3.SA', 'IVVB11.SA', 'TAEE11.SA', 'WEGE3.SA', 'LREN3.SA',
    'BOVA11.SA', 'SMAL11.SA', 'PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'HGLG11.SA'
]

index_ticker = '^BVSP'
start_date = '2016-07-01'
end_date = '2026-07-01'  

print("Baixando os dados dos ativos e do Ibovespa...")

data_tickers = yf.download(
    tickers,
    start=start_date,
    end=end_date,
    auto_adjust=True
)['Close']

data_index = yf.download(
    index_ticker,
    start=start_date,
    end=end_date,
    auto_adjust=True
)['Close']

data = pd.concat([data_tickers, data_index], axis=1).rename(
    columns={index_ticker: 'IBOVESPA'}
)

retornos = np.log(data / data.shift(1)).dropna()


# 2. CONSTRUÇÃO DOS INDICADORES E DAS VARIÁVEIS-ALVO

print("Calculando os indicadores técnicos...")

ativos_para_features = tickers + ['IBOVESPA']
features_df = pd.DataFrame(index=data.index)

# Médias móveis
janelas_sma = [20, 50]

for janela in janelas_sma:
    for ativo in ativos_para_features:
        sma = data[ativo].rolling(window=janela).mean()
        features_df[f'SMA_{janela}_{ativo}'] = (data[ativo] / sma) - 1

# Índice de Força Relativa (RSI)
janela_rsi = 14

for ativo in ativos_para_features:
    delta = data[ativo].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.ewm(
        com=janela_rsi - 1,
        min_periods=janela_rsi
    ).mean()

    avg_loss = loss.ewm(
        com=janela_rsi - 1,
        min_periods=janela_rsi
    ).mean()

    rs = avg_gain / avg_loss
    features_df[f'RSI_{ativo}'] = 100 - (100 / (1 + rs))

# Volatilidade histórica anualizada
janela_vol = 21

for ativo in ativos_para_features:
    vol_diaria = retornos[ativo].rolling(window=janela_vol).std()
    features_df[f'VOL_{ativo}'] = vol_diaria * np.sqrt(252)

# O retorno logarítmico do próximo pregão é a variável-alvo de cada ativo.
targets_para_ml = tickers

target_df = retornos[targets_para_ml].shift(-1).rename(
    columns=lambda x: f'TARGET_{x}'
)

# Seleciona o pregão mais recente com indicadores completos para a previsão.
features_validas = features_df.dropna()
features_hoje = features_validas.iloc[[-1]]
data_hoje = features_hoje.index[0]

# Reúne os indicadores e os retornos-alvo conhecidos para treino e teste.
dataset_ml = pd.concat([features_df, target_df], axis=1).dropna()

print(f"Pregão de referência para a previsão: {data_hoje.date()}")
print(f"Última data de referência no conjunto de treino e teste: {dataset_ml.index[-1].date()}")


# 3. DIVISÃO TEMPORAL E PADRONIZAÇÃO DOS DADOS

X = dataset_ml.drop(
    columns=[col for col in dataset_ml.columns if 'TARGET_' in col]
)

y = dataset_ml[
    [col for col in dataset_ml.columns if 'TARGET_' in col]
]

# Aplica a divisão inicial de 80% para treino e 20% para teste.
split_index = int(0.8 * len(X))

if len(X) - split_index < 100:
    split_index = len(X) - 100

X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

features_hoje = features_hoje[X.columns]

print("\n=== DIVISÃO TEMPORAL ===")
print(f"Treino: {len(X_train)} observações ({len(X_train) / len(X) * 100:.2f}%)")
print(f"Teste: {len(X_test)} observações ({len(X_test) / len(X) * 100:.2f}%)")
print(f"Período de treino: {X_train.index[0].date()} até {X_train.index[-1].date()}")
print(f"Período de teste: {X_test.index[0].date()} até {X_test.index[-1].date()}")

# Ajusta a padronização com os dados de treino.
scaler_X = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
features_hoje_scaled = scaler_X.transform(features_hoje)


# 4. CONFIGURAÇÃO DA BUSCA DE HIPERPARÂMETROS COM OPTUNA

optuna.logging.set_verbosity(optuna.logging.WARNING)

N_TRIALS = 100
N_SPLITS_OPTUNA = 5


def objective_xgb(trial, X_treino, y_treino):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 150),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'objective': 'reg:squarederror',
        'random_state': 42,
        'n_jobs': -1
    }

    tscv = TimeSeriesSplit(n_splits=N_SPLITS_OPTUNA)
    erros = []

    for train_idx, val_idx in tscv.split(X_treino):
        X_tr, X_val = X_treino[train_idx], X_treino[val_idx]
        y_tr, y_val = y_treino[train_idx], y_treino[val_idx]

        model = XGBRegressor(**params)
        model.fit(X_tr, y_tr)

        pred_val = model.predict(X_val)
        mse = mean_squared_error(y_val, pred_val)
        erros.append(mse)

    return np.mean(erros)


# 5. AJUSTE, AVALIAÇÃO E PREVISÃO COM XGBOOST

print("\nAjustando e avaliando os modelos XGBoost por ativo...")

mu_predito = []
metricas_modelos = []
melhores_parametros = []

for i, ativo in enumerate(tickers, start=1):
    print(f"\n[{i}/{len(tickers)}] Ativo: {ativo}")

    target_col = f'TARGET_{ativo}'

    y_train_ativo = y_train[target_col].values
    y_test_ativo = y_test[target_col].values

    # Busca os hiperparâmetros por validação temporal no conjunto de treino.
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    study.optimize(
        lambda trial: objective_xgb(
            trial,
            X_train_scaled,
            y_train_ativo
        ),
        n_trials=N_TRIALS
    )

    best_params = study.best_params.copy()

    print(f"Melhores hiperparâmetros: {best_params}")

    melhores_parametros.append({
        'Ativo': ativo,
        **best_params,
        'MSE_Validacao_Optuna': study.best_value
    })

    # Treina o modelo final com todo o conjunto de treino.
    model = XGBRegressor(
        **best_params,
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train_scaled, y_train_ativo)

    # Avalia o modelo no conjunto de teste.
    pred_test = model.predict(X_test_scaled)

    rmse = np.sqrt(mean_squared_error(y_test_ativo, pred_test))
    mae = mean_absolute_error(y_test_ativo, pred_test)
    r2 = r2_score(y_test_ativo, pred_test)

    acerto_direcional = np.mean(
        np.sign(pred_test) == np.sign(y_test_ativo)
    )
    acerto_direcional_pct = acerto_direcional * 100

    # Prevê o retorno do próximo pregão.
    pred_amanha = model.predict(features_hoje_scaled)[0]
    mu_predito.append(pred_amanha)

    # Calcula a média dos retornos logarítmicos diários da série disponível.
    media_historica = retornos[ativo].mean()

    metricas_modelos.append({
        'Ativo': ativo,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'Acerto_Direcional_%': acerto_direcional_pct,
        'Media_Historica_Diaria': media_historica,
        'Mu_Predito': pred_amanha
    })


# 6. EXPORTAÇÃO DAS MÉTRICAS E DOS HIPERPARÂMETROS

df_metricas = pd.DataFrame(metricas_modelos)

print("\n=== AVALIAÇÃO DOS MODELOS XGBOOST NO CONJUNTO DE TESTE ===")
print(df_metricas.to_string(index=False))

df_metricas.to_csv(
    'avaliacao_xgboost_optuna.csv',
    index=False
)

df_melhores_parametros = pd.DataFrame(melhores_parametros)

print("\n=== MELHORES HIPERPARÂMETROS POR ATIVO ===")
print(df_melhores_parametros.to_string(index=False))

df_melhores_parametros.to_csv(
    'melhores_parametros_xgboost.csv',
    index=False
)

# Retornos previstos utilizados na otimização.
mu_final = np.array(mu_predito)


# 7. ESTIMAÇÃO DA MATRIZ DE COVARIÂNCIA POR LEDOIT–WOLF

print("\nEstimando a matriz de covariância dos retornos diários...")

lw = LedoitWolf().fit(retornos[targets_para_ml])
cov_final = lw.covariance_


# 8. EXPORTAÇÃO DOS INSUMOS PARA A OTIMIZAÇÃO DE CARTEIRAS

dados_para_otimizacao = {
    'mu': pd.Series(mu_final, index=tickers),
    'cov': pd.DataFrame(cov_final, index=tickers, columns=tickers),
    'tickers': tickers,
    'data_referencia': data_hoje
}

with open('dados_preparados.pkl', 'wb') as f:
    pickle.dump(dados_para_otimizacao, f)
print(
    "\nParte 1 concluída. "
    "Os insumos para a otimização foram salvos em 'dados_preparados.pkl'."
)