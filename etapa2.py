import warnings
warnings.filterwarnings('ignore')
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.indicators.hv import Hypervolume
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from scipy.stats import mannwhitneyu


try:
    from mola.mola import multi_objective_lichtenberg_algorithm
except ImportError:
    from mola import multi_objective_lichtenberg_algorithm


# 1. PARÂMETROS GERAIS

LIMIAR_CARDINALIDADE = 0.05  # Conta os ativos com peso acima de 5%.
LIMIAR_TXT = f'{LIMIAR_CARDINALIDADE * 100:.0f}%'
REF_NORMALIZADO = 1.1  # Coordenada de referência em cada objetivo normalizado.
PASTA = 'graficos_resultados_7pct'
os.makedirs(PASTA, exist_ok=True)


SEMENTES = [10, 25, 42, 99, 123, 555, 777, 888, 2024, 2026]


# 2. CARREGAMENTO DOS INSUMOS DA PARTE 1

if not os.path.isfile('dados_preparados.pkl'):
    raise FileNotFoundError('Execute a Parte 1 para gerar dados_preparados.pkl.')
with open('dados_preparados.pkl', 'rb') as arq:
    dados = pickle.load(arq)
tickers = list(dados['tickers'])
n_ativos = len(tickers)
mu_final = np.asarray(dados['mu'].loc[tickers] if isinstance(dados['mu'], pd.Series) else dados['mu'], dtype=float).reshape(-1)
cov_final = np.asarray(dados['cov'].loc[tickers, tickers] if isinstance(dados['cov'], pd.DataFrame) else dados['cov'], dtype=float)
if mu_final.shape != (n_ativos,) or cov_final.shape != (n_ativos, n_ativos):
    raise ValueError('Dimensões de mu/cov não correspondem aos tickers.')
if not (np.all(np.isfinite(mu_final)) and np.all(np.isfinite(cov_final))):
    raise ValueError('mu ou cov contêm valores não finitos.')


# 3. FORMULAÇÃO DOS OBJETIVOS DA OTIMIZAÇÃO

def normalizar_pesos(x):
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size != n_ativos or not np.all(np.isfinite(x)):
        raise ValueError('Vetor de decisão inválido.')
    soma = x.sum()
    return x / soma if soma != 0 else np.ones(n_ativos) / n_ativos


def fhandle_portfolio(x):
    pesos = normalizar_pesos(x)
    retorno_anual = np.dot(pesos, mu_final) * 252
    variancia = float(pesos @ cov_final @ pesos)
    risco_anual = np.sqrt(max(variancia, 0.0)) * np.sqrt(252)
    qtd_ativos = float(np.sum(pesos > LIMIAR_CARDINALIDADE))
    return np.array([-retorno_anual, risco_anual, qtd_ativos], dtype=float)


def fnonlin_portfolio(x):
    return [0.0], [0.0]


class PortfolioProblemPymoo(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var=n_ativos, n_obj=3, n_constr=0, xl=0.0, xu=1.0)


    def _evaluate(self, x, out, *args, **kwargs):
        out['F'] = fhandle_portfolio(x)


problem_pymoo = PortfolioProblemPymoo()


# 4. VALIDAÇÃO DAS SOLUÇÕES

def preparar_solucoes(X, F=None):
    if X is None:
        return np.empty((0, 3)), np.empty((0, n_ativos))
    X = np.asarray(X, dtype=float)
    if X.size == 0:
        return np.empty((0, 3)), np.empty((0, n_ativos))
    X = np.atleast_2d(X)
    if X.shape[1] != n_ativos:
        raise ValueError(f'X tem dimensão {X.shape}; esperada (*, {n_ativos}).')
    if F is not None and np.size(F):
        F = np.asarray(F, dtype=float)
        F = np.atleast_2d(F)
        if F.shape != (len(X), 3):
            raise ValueError(f'F e X do algoritmo não estão alinhados: {F.shape} versus {X.shape}.')
    # Recalcula os objetivos (F) a partir dos pesos das carteiras (X).

    validos_x = np.all(np.isfinite(X), axis=1) & np.all(X >= -1e-10, axis=1) & np.all(X <= 1 + 1e-10, axis=1)
    X = X[validos_x]
    if not len(X):
        return np.empty((0, 3)), np.empty((0, n_ativos))
    F_calc = np.array([fhandle_portfolio(x) for x in X])
    # Aplica os mesmos critérios de validade às soluções dos dois algoritmos.
    mascara = np.all(np.isfinite(F_calc), axis=1) & (F_calc[:, 0] < 1e5)
    return F_calc[mascara], X[mascara]


# 5. EXECUÇÃO DO NSGA-II E DO MOLA
# Registra cada rodada, inclusive quando um dos algoritmos falha.
# Registra cada rodada, inclusive quando um dos algoritmos falha.
frentes_nsga2_limpas, pesos_nsga2_limpas = [], []
frentes_mola_limpas, pesos_mola_limpas = [], []
registro_rodadas = []
ultimas_frentes = {}


print('\nExecutando 10 rodadas dos algoritmos...')
for rodada, semente in enumerate(SEMENTES, start=1):
    print(f'Rodada {rodada}/10 | semente {semente}')
    np.random.seed(semente)
    F_nsga, X_nsga = np.empty((0, 3)), np.empty((0, n_ativos))
    F_mola, X_mola = np.empty((0, 3)), np.empty((0, n_ativos))
    erro_nsga, erro_mola = '', ''
    try:
        resultado = minimize(problem_pymoo, NSGA2(pop_size=100,sampling=FloatRandomSampling()),
                             get_termination('n_gen', 150), seed=semente, verbose=False)
        F_nsga, X_nsga = preparar_solucoes(resultado.X, resultado.F)
        if not len(F_nsga):
            erro_nsga = 'Nenhuma solução válida'
    except Exception as exc:
        erro_nsga = f'{type(exc).__name__}: {exc}'
        print(f'  AVISO: NSGA-II falhou: {erro_nsga}')


    try:
        XPF, FVALPF = multi_objective_lichtenberg_algorithm(
            fhandle=fhandle_portfolio, n_objectives=3, d=n_ativos, pop=100,
            LB=np.zeros(n_ativos), UB=np.ones(n_ativos), ref=0,
            n_iter=150, ngrid=25, Nr=250, IntCon=0,
            fnonlin=fnonlin_portfolio
        )
        F_mola, X_mola = preparar_solucoes(XPF, FVALPF)
        if not len(F_mola):
            erro_mola = 'Nenhuma solução válida'
    except Exception as exc:
        erro_mola = f'{type(exc).__name__}: {exc}'
        print(f'  AVISO: MOLA falhou: {erro_mola}')


    frentes_nsga2_limpas.append(F_nsga)
    pesos_nsga2_limpas.append(X_nsga)
    frentes_mola_limpas.append(F_mola)
    pesos_mola_limpas.append(X_mola)
    registro_rodadas.append(dict(rodada=rodada, semente=semente,
                                 n_nsga=len(F_nsga), n_mola=len(F_mola),
                                 erro_nsga=erro_nsga, erro_mola=erro_mola))
    # Guarda a última frente válida de cada algoritmo para os gráficos individuais.
    if len(F_nsga):
        ultimas_frentes['NSGA-II'] = F_nsga
    if len(F_mola):
        ultimas_frentes['MOLA'] = F_mola


pd.DataFrame(registro_rodadas).to_csv(os.path.join(PASTA, 'registro_rodadas.csv'), index=False)
if not any(len(f) for f in frentes_nsga2_limpas + frentes_mola_limpas):
    raise RuntimeError('Nenhuma execução gerou soluções válidas.')


# 6. CÁLCULO DO HIPERVOLUME NORMALIZADO

# Coloca os três objetivos em uma escala comum para calcular o hipervolume.


todas_frentes_validas = [f for f in frentes_nsga2_limpas + frentes_mola_limpas if len(f)]
todos_pontos_reais = np.vstack(todas_frentes_validas)


f_min = np.min(todos_pontos_reais, axis=0)
f_max = np.max(todos_pontos_reais, axis=0)
amplitude = np.where((f_max - f_min) > 0, f_max - f_min, 1.0)  # Usa amplitude unitária para objetivos constantes.


def normalizar_objetivos(F):
    return (np.asarray(F, dtype=float) - f_min) / amplitude


ref_point = np.full(3, REF_NORMALIZADO)
hv_indicator = Hypervolume(ref_point=ref_point)
hv_nsga2 = [float(hv_indicator.do(normalizar_objetivos(f))) if len(f) else np.nan for f in frentes_nsga2_limpas]
hv_mola = [float(hv_indicator.do(normalizar_objetivos(f))) if len(f) else np.nan for f in frentes_mola_limpas]


pd.DataFrame({'objetivo': ['-retorno_anual', 'risco_anual', f'qtd_ativos_>{LIMIAR_TXT}'],
              'min': f_min, 'max': f_max}).to_csv(
                  os.path.join(PASTA, 'limites_normalizacao_hv.csv'), index=False)


def resumo_hv(nome, valores):
    val = np.asarray(valores, dtype=float)
    val = val[np.isfinite(val)]


    if not len(val):
        print(f'{nome}: nenhuma rodada válida')
    else:
        q1 = np.percentile(val, 25)
        mediana = np.median(val)
        q3 = np.percentile(val, 75)


        print(
            f'{nome}: '
            f'média {val.mean():.5f} | '
            f'mediana {mediana:.5f} | '
            f'Q1 {q1:.5f} | '
            f'Q3 {q3:.5f} | '
            f'desvio padrão {val.std():.5f} | '
            f'{len(val)}/{len(SEMENTES)} rodadas válidas'
        )
print('\n=== RESULTADOS DO HIPERVOLUME (objetivos normalizados, ref = '
      f'{REF_NORMALIZADO}) ===')
resumo_hv('NSGA-II', hv_nsga2)
resumo_hv('MOLA', hv_mola)


# 7. COMPARAÇÃO DO HIPERVOLUME PELO TESTE DE MANN–WHITNEY

from scipy.stats import mannwhitneyu


hv_n = np.asarray(hv_nsga2, dtype=float)
hv_m = np.asarray(hv_mola, dtype=float)


# Seleciona os valores de hipervolume das rodadas válidas.
hv_n = hv_n[np.isfinite(hv_n)]
hv_m = hv_m[np.isfinite(hv_m)]


u_stat, p_valor = mannwhitneyu(
    hv_n,
    hv_m,
    alternative='two-sided'
)


print('\n=== TESTE DE MANN-WHITNEY PARA O HIPERVOLUME ===')
print(f'U = {u_stat:.3f}')
print(f'p-valor = {p_valor:.5f}')


if p_valor < 0.05:
    print('Há diferença estatisticamente significativa entre os algoritmos (α = 0,05).')
else:
    print('Não foi detectada diferença estatisticamente significativa entre os algoritmos (α = 0,05).')


pd.DataFrame({'rodada': range(1, len(SEMENTES)+1), 'semente': SEMENTES,
              'hv_nsga2': hv_nsga2, 'hv_mola': hv_mola}).to_csv(
                  os.path.join(PASTA, 'hipervolume_por_rodada.csv'), index=False)


plt.figure(figsize=(10, 4))
plt.plot(range(1, 11), hv_nsga2, marker='o', color='#d62728', label='NSGA-II')
plt.plot(range(1, 11), hv_mola, marker='s', color='#2ca02c', linestyle='--', label='MOLA')
plt.title('Estabilidade estocástica do hipervolume (objetivos normalizados)')
plt.xlabel('Rodada'); plt.ylabel('Hipervolume normalizado'); plt.xticks(range(1, 11))
plt.grid(linestyle=':', alpha=0.6); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(PASTA, 'comparativo_estabilidade_hypervolume.png'), dpi=300)
plt.close()


# 8. GRÁFICOS DAS FRENTES DAS ÚLTIMAS RODADAS VÁLIDAS

fig, axs = plt.subplots(1, 2, figsize=(16, 6))
for ax, nome, cmap in zip(axs, ['NSGA-II', 'MOLA'], ['viridis', 'plasma']):
    f = ultimas_frentes.get(nome)
    if f is None:
        ax.text(0.5, 0.5, 'Sem execução válida', ha='center', va='center', transform=ax.transAxes)
    else:
        sc = ax.scatter(f[:, 1], -f[:, 0], c=f[:, 2], cmap=cmap, s=50,
                        edgecolors='black', linewidths=.5, alpha=.85)
        fig.colorbar(sc, ax=ax, pad=.02).set_label(f'Ativos com peso > {LIMIAR_TXT}')
    ax.set_title(f'Frente de Pareto: {nome}')
    ax.set_xlabel('Risco anualizado'); ax.set_ylabel('Retorno esperado anualizado')
    ax.grid(linestyle=':', alpha=.5)
plt.tight_layout()
plt.savefig(os.path.join(PASTA, 'frente_pareto_risco_retorno_2D.png'), dpi=300)
plt.close()


fig = plt.figure(figsize=(16, 7))
for i, (nome, cmap) in enumerate([('NSGA-II', 'viridis'), ('MOLA', 'plasma')], start=1):
    ax = fig.add_subplot(1, 2, i, projection='3d')
    f = ultimas_frentes.get(nome)
    if f is not None:
        sc = ax.scatter(f[:, 1], -f[:, 0], f[:, 2], c=f[:, 2], cmap=cmap,
                        s=45, edgecolors='black', linewidths=.3, alpha=.9)
        fig.colorbar(sc, ax=ax, shrink=.5, aspect=15, pad=.1)
    ax.set_title(f'Frente de Pareto 3D: {nome}')
    ax.set_xlabel('Risco'); ax.set_ylabel('Retorno'); ax.set_zlabel(f'Nº ativos > {LIMIAR_TXT}')
    ax.view_init(elev=20, azim=130)
plt.tight_layout()
plt.savefig(os.path.join(PASTA, 'frente_pareto_risco_retorno_3D.png'), dpi=300)
plt.close()


# 9. CONSTRUÇÃO DA FRENTE GLOBAL DE PARETO

def obter_frente_global_combinada_com_origem(frentes_mola, pesos_mola, frentes_nsga2, pesos_nsga2):
    pontos, pesos, origens, rodadas = [], [], [], []
    for codigo, frentes, vetores in [(0, frentes_mola, pesos_mola),
                                     (1, frentes_nsga2, pesos_nsga2)]:
        if len(frentes) != len(SEMENTES) or len(vetores) != len(SEMENTES):
            raise ValueError('Histórico incompleto de rodadas.')
        for rodada, (f, x) in enumerate(zip(frentes, vetores), start=1):
            if len(f) != len(x):
                raise ValueError(f'Desalinhamento F/X em {codigo}, rodada {rodada}.')
            if len(f):
                pontos.append(f)
                pesos.append(x)
                origens.extend([codigo] * len(f))
                rodadas.extend([rodada] * len(f))
    if not pontos:
        raise ValueError('Não há pontos para a frente combinada.')
    F = np.vstack(pontos)
    X = np.vstack(pesos)
    origens = np.asarray(origens, dtype=int)
    rodadas = np.asarray(rodadas, dtype=int)
    indices = np.asarray(NonDominatedSorting().do(F, only_non_dominated_front=True), dtype=int)
    return F[indices], origens[indices], X[indices], rodadas[indices]


pf_gmol_combinada, algoritmo_origem, pesos_pf_combinada, rodada_origem = (
    obter_frente_global_combinada_com_origem(frentes_mola_limpas, pesos_mola_limpas,
                                            frentes_nsga2_limpas, pesos_nsga2_limpas)
)


# 10. CONTRIBUIÇÃO DOS ALGORITMOS PARA A FRENTE GLOBAL

# Identifica soluções coincidentes pelos objetivos arredondados a oito casas decimais.

# Preserva as soluções duplicadas nos registros brutos para conferência.
# Calcula os percentuais a partir dos grupos de soluções coincidentes.
origem_por_chave = {}
representante_por_chave = {}
for idx, ponto in enumerate(pf_gmol_combinada):
    chave = tuple(np.round(ponto, decimals=8))
    origem_por_chave.setdefault(chave, set()).add(int(algoritmo_origem[idx]))
    representante_por_chave.setdefault(chave, idx)


n_mola = sum(v == {0} for v in origem_por_chave.values())
n_nsga = sum(v == {1} for v in origem_por_chave.values())
n_comp = sum(v == {0, 1} for v in origem_por_chave.values())
total_unicos = len(origem_por_chave)
print('\n=== CONTRIBUIÇÃO PARA A FRENTE GLOBAL ===')
for nome, n in [('MOLA exclusivas', n_mola), ('NSGA-II exclusivas', n_nsga), ('Compartilhadas', n_comp)]:
    print(f'{nome}: {n} ({100*n/total_unicos:.2f}%)')
print(f'Total de soluções distintas (8 casas): {total_unicos}')
df_contribuicao = pd.DataFrame({
    'origem': ['MOLA (exclusivas)', 'NSGA-II (exclusivas)', 'Compartilhadas'],
    'solucoes': [n_mola, n_nsga, n_comp],
    'percentual': [100*n_mola/total_unicos, 100*n_nsga/total_unicos, 100*n_comp/total_unicos]
})
df_contribuicao.to_csv(os.path.join(PASTA, 'contribuicao_frente_global.csv'), index=False)


fig, ax = plt.subplots(figsize=(9, 5))
barras = ax.bar(df_contribuicao['origem'], df_contribuicao['percentual'],
                color=['#2ca02c', '#d62728', '#7f7f7f'])
for b, n in zip(barras, df_contribuicao['solucoes']):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+.5, str(n), ha='center')
ax.set_ylim(0, max(10, df_contribuicao['percentual'].max()*1.2))
ax.set_ylabel('Participação na frente global (%)')
ax.set_title('Contribuição dos algoritmos à frente global de Pareto')
ax.grid(axis='y', linestyle=':', alpha=.4)
plt.tight_layout()
plt.savefig(os.path.join(PASTA, 'contribuicao_frente_global.png'), dpi=300)
plt.close()


# 11. SELEÇÃO DA CARTEIRA PELO TOPSIS

ponto_ideal = np.min(pf_gmol_combinada, axis=0)
ponto_nadir = np.max(pf_gmol_combinada, axis=0)


def executar_topsis(frente, p_ideal, p_nadir, pesos=(0.4, 0.4, 0.2)):
    # Protege a normalização dos critérios contra divisões por zero.
    norma = np.linalg.norm(frente, axis=0)
    norma = np.where(norma > 0, norma, 1.0)
    pesos = np.asarray(pesos, dtype=float)
    v_frente = frente / norma * pesos
    v_ideal = p_ideal / norma * pesos
    v_nadir = p_nadir / norma * pesos
    d_ideal = np.linalg.norm(v_frente - v_ideal, axis=1)
    d_nadir = np.linalg.norm(v_frente - v_nadir, axis=1)
    divisor = d_ideal + d_nadir
    score = np.divide(d_nadir, divisor, out=np.full_like(divisor, .5), where=divisor > 0)
    return int(np.argmax(score)), score


melhor_idx, scores = executar_topsis(pf_gmol_combinada, ponto_ideal, ponto_nadir)
carteira_escolhida = pf_gmol_combinada[melhor_idx]
pesos_carteira_escolhida = pesos_pf_combinada[melhor_idx]
algoritmo_escolhido = 'MOLA' if algoritmo_origem[melhor_idx] == 0 else 'NSGA-II'
rodada_escolhida = int(rodada_origem[melhor_idx])
pesos_normalizados_topsis = normalizar_pesos(pesos_carteira_escolhida)


print('\n=== FRENTE GLOBAL E CARTEIRA TOPSIS ===')
print(f'Número de pontos brutos não dominados: {len(pf_gmol_combinada)}')
print(f'Ideal: {ponto_ideal} | Nadir: {ponto_nadir}')
print(f'Origem: {algoritmo_escolhido}, rodada: {rodada_escolhida}, semente: {SEMENTES[rodada_escolhida - 1]}')
print(f'Retorno anualizado: {-carteira_escolhida[0]*100:.2f}%')
print(f'Risco anualizado: {carteira_escolhida[1]*100:.2f}%')
print(f'Ativos >{LIMIAR_TXT}: {int(carteira_escolhida[2])}')
print(f'Score TOPSIS: {scores[melhor_idx]:.6f}')


df_carteira_topsis = pd.DataFrame({
    'Ativo': tickers, 'Peso': pesos_normalizados_topsis,
    'Peso (%)': pesos_normalizados_topsis * 100
}).sort_values('Peso', ascending=False).reset_index(drop=True)
print('\n=== COMPOSIÇÃO DA CARTEIRA ===')
print(df_carteira_topsis[['Ativo', 'Peso (%)']].to_string(index=False))
print(f'Total: {df_carteira_topsis["Peso (%)"].sum():.4f}%')
print(f'\n=== ATIVOS COM PESO > {LIMIAR_TXT} ===')
print(df_carteira_topsis.loc[df_carteira_topsis['Peso'] > LIMIAR_CARDINALIDADE, ['Ativo', 'Peso (%)']].to_string(index=False))
df_carteira_topsis.to_csv(os.path.join(PASTA, 'composicao_carteira_topsis.csv'),
                         index=False, sep=';', decimal=',')


# Exporta os registros das soluções da frente global para conferência.
linhas_frente = []
for i, (f, x, origem, rodada) in enumerate(zip(pf_gmol_combinada, pesos_pf_combinada,
                                               algoritmo_origem, rodada_origem)):
    item = {'indice': i, 'algoritmo': 'MOLA' if origem == 0 else 'NSGA-II',
            'rodada': int(rodada), 'semente': SEMENTES[int(rodada)-1],
            'retorno_anual': -float(f[0]), 'risco_anual': float(f[1]),
            f'ativos_acima_{LIMIAR_TXT.replace("%", "pct")}': int(f[2]),
            'score_topsis': float(scores[i])}
    item.update({f'peso_{t}': w for t, w in zip(tickers, normalizar_pesos(x))})
    linhas_frente.append(item)
pd.DataFrame(linhas_frente).to_csv(os.path.join(PASTA, 'frente_global_detalhada.csv'), index=False)


# 12. PROJEÇÃO DA FRENTE GLOBAL NO PLANO RISCO-RETORNO

fig, ax = plt.subplots(figsize=(11, 7))
ax.set_facecolor('#fefefe')
ax.grid(True, linestyle='--', alpha=.4, zorder=1)
for codigo, nome, mapa in [(1, 'NSGA-II', 'viridis_r'), (0, 'MOLA', 'plasma')]:
    mask = algoritmo_origem == codigo
    if np.any(mask):
        f = pf_gmol_combinada[mask]
        ax.scatter(f[:, 1], -f[:, 0], c=-f[:, 0], cmap=mapa, marker='o',
                   s=75, alpha=.9, edgecolors='#0f172a', linewidths=.9, zorder=3,
                   label=f'Soluções {nome}')
ax.scatter(ponto_ideal[1], -ponto_ideal[0], c='#ff4500', marker='^', s=240,
           edgecolors='black', zorder=5, label='Ponto ideal (projeção)')
ax.scatter(ponto_nadir[1], -ponto_nadir[0], c='#475569', marker='v', s=240,
           edgecolors='black', zorder=5, label='Ponto de Nadir (projeção)')
ax.scatter(carteira_escolhida[1], -carteira_escolhida[0], c='#ffea00', marker='*',
           s=650, edgecolors='black', linewidths=2.5, zorder=10, label='Solução TOPSIS')
rotulo = 'Solução selecionada pelo TOPSIS'


# Define o deslocamento da anotação em função da amplitude dos dados.
risco_amp = max(float(np.ptp(pf_gmol_combinada[:, 1])), .01)
ret_amp = max(float(np.ptp(-pf_gmol_combinada[:, 0])), .01)
ax.annotate(rotulo, xy=(carteira_escolhida[1], -carteira_escolhida[0]),
            xytext=(carteira_escolhida[1] + .10*risco_amp,
                    -carteira_escolhida[0] - .15*ret_amp),
            arrowprops=dict(arrowstyle='->', color='#ef4444'),
            fontsize=9, fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=.6', fc='#1e293b', ec='black', alpha=.95), zorder=11)
ax.set_title('TOPSIS na frente global combinada (projeção 2D)')
ax.set_xlabel('Risco anualizado'); ax.set_ylabel('Retorno esperado anualizado')
ax.legend(loc='best', frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(PASTA, 'selecao_carteira_topsis.png'), dpi=350, bbox_inches='tight')
plt.close()


# 13. GRÁFICO DA COMPOSIÇÃO DA CARTEIRA

df_plot = df_carteira_topsis.sort_values('Peso (%)')
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(df_plot['Ativo'], df_plot['Peso (%)'])
ax.set_title('Composição da carteira selecionada pelo TOPSIS')
ax.set_xlabel('Participação na carteira (%)'); ax.set_ylabel('Ativo')
ax.grid(axis='x', linestyle=':', alpha=.5)
for pos, valor in enumerate(df_plot['Peso (%)']):
    ax.text(valor, pos, f' {valor:.2f}%', va='center', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(PASTA, 'composicao_carteira_topsis.png'), dpi=300)
plt.close()


print(f'\nProcesso concluído. Arquivos salvos em {PASTA}/')
