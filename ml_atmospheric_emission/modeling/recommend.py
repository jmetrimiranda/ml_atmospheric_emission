import numpy as np
import pandas as pd

from ml_atmospheric_emission.config import MODELO_ATUAL, PREVISOR_POR_CABECA, RECOMENDACAO_VALIDADA, VARIAVEIS_CONTROLE
from ml_atmospheric_emission.dataset import load_interim, load_processed
from ml_atmospheric_emission.features import create_model_features
from ml_atmospheric_emission.modeling.predict import load_head_models


def predict_risk(modelos, X):
    #Risk of plume: probability of a single model, or the MEAN of a list of models (one model per head)
    #All the models must use the same columns (true for base_estado: the columns do not depend on the head)
    if not isinstance(modelos, list):
        modelos = [modelos]
    soma = 0
    for model in modelos:
        soma = soma + model.predict_proba(X)[:, 1]
    return soma / len(modelos)


def create_limits(ds, variaveis_controle, percentil_min=5, percentil_max=95):
    #Allowed range of each controllable variable: only values the plant already operated
    limites = {}
    for variavel in variaveis_controle:
        minimo = np.nanpercentile(ds[variavel], percentil_min)
        maximo = np.nanpercentile(ds[variavel], percentil_max)
        desvio = ds[variavel].std()
        limites[variavel] = (minimo, maximo, desvio)
    return limites


def create_support(ds, variaveis_controle, percentil=99):
    #Mahalanobis distance: measures how unusual a combination of tags is,
    #considering that the tags move together (correlated)
    X = ds[variaveis_controle].dropna().values
    media = X.mean(axis=0)
    cov_inv = np.linalg.inv(np.cov(X, rowvar=False))
    diferenca = X - media
    distancias = np.sqrt(np.sum(diferenca @ cov_inv * diferenca, axis=1))
    suporte = {
        'media': media,
        'cov_inv': cov_inv,
        'distancia_maxima': np.percentile(distancias, percentil)
    }
    return suporte


def distance_to_history(valores, suporte):
    diferenca = np.array(valores) - suporte['media']
    return np.sqrt(np.sum(diferenca @ suporte['cov_inv'] * diferenca, axis=-1))


def recommend_actions(model, linha, features, variaveis_controle, limites, suporte, passo_maximo=0.5, n_valores=11, n_rodadas=2):
    #Searches small changes in the controllable variables that lower the predicted risk
    #model: one model, or a list of models (one per head): the risk is the mean of them (predict_risk)
    #Rules: only controllable variables change, each one at most passo_maximo standard deviations,
    #inside the historical range, and the combination of tags must look like a real operation
    linha_original = linha[features].astype(float)
    risco_original = predict_risk(model, linha_original.to_frame().T)[0]
    recomendacao = {
        'risco_atual': round(risco_original, 3),
        'risco_recomendado': round(risco_original, 3),
        'mudancas': [],
        'linha_recomendada': linha_original.copy()
    }
    #A tag without value (NaN): no recommendation for this moment
    if linha_original[variaveis_controle].isna().any():
        return recomendacao

    melhor = linha_original.copy()
    melhor_risco = risco_original
    distancia_original = distance_to_history(linha_original[variaveis_controle].values, suporte)
    distancia_permitida = max(suporte['distancia_maxima'], distancia_original)

    for rodada in range(n_rodadas):
        for variavel in variaveis_controle:
            valor_original = linha_original[variavel]
            minimo, maximo, desvio = limites[variavel]
            inferior = max(minimo, valor_original - passo_maximo*desvio)
            superior = min(maximo, valor_original + passo_maximo*desvio)
            if inferior >= superior:
                continue
            #All candidate values for this variable, keeping the best values found for the others
            candidatos = pd.DataFrame([melhor] * n_valores).reset_index(drop=True)
            candidatos[variavel] = np.linspace(inferior, superior, n_valores)
            riscos = predict_risk(model, candidatos[features])
            distancias = distance_to_history(candidatos[variaveis_controle].values, suporte)
            cont = 0
            for risco in riscos:
                if risco < melhor_risco and distancias[cont] <= distancia_permitida:
                    melhor_risco = risco
                    melhor = candidatos.iloc[cont].copy()
                cont = cont+1

    #Only the variables that really changed
    mudancas = []
    for variavel in variaveis_controle:
        if not np.isclose(melhor[variavel], linha_original[variavel]):
            if melhor[variavel] > linha_original[variavel]:
                direcao = 'subir'
            else:
                direcao = 'baixar'
            mudancas.append({
                'variavel': variavel,
                'valor_atual': round(linha_original[variavel], 3),
                'valor_recomendado': round(melhor[variavel], 3),
                'direcao': direcao
            })
    recomendacao['risco_recomendado'] = round(melhor_risco, 3)
    recomendacao['mudancas'] = mudancas
    recomendacao['linha_recomendada'] = melhor
    return recomendacao


def print_recommendation(data, recomendacao):
    print('Momento:', data)
    print('Risco de pluma previsto agora:', recomendacao['risco_atual'])
    print('Risco com a recomendacao:', recomendacao['risco_recomendado'])
    if len(recomendacao['mudancas']) == 0:
        print('Nenhuma mudanca dentro das regras reduz o risco')
    for mudanca in recomendacao['mudancas']:
        print(' ', mudanca['direcao'], mudanca['variavel'], 'de', mudanca['valor_atual'], 'para', mudanca['valor_recomendado'])
    if 'risco_por_cabeca' in recomendacao:
        print('Risco de cada cabeca (agora -> com a recomendacao):')
        for minutos in recomendacao['risco_por_cabeca']:
            antes, depois = recomendacao['risco_por_cabeca'][minutos]
            print('   ', minutos, 'min:', antes, '->', depois)
    if 'validada' in recomendacao and not recomendacao['validada']:
        print('ATENCAO: recomendacao NAO validada (notebook 6.0). E uma hipotese para testar com a operacao, nao uma ordem')
    print()


def opposite_change(linha_original, linha_recomendada, variaveis_controle):
    #Control: the same change, in the opposite direction (same variables, same size)
    contraria = linha_original.copy()
    for variavel in variaveis_controle:
        mudanca = linha_recomendada[variavel] - linha_original[variavel]
        contraria[variavel] = linha_original[variavel] - mudanca
    return contraria


def validate_recommendations(model_rec, model_val, ds_periodo, features, variaveis_controle, n_momentos=150, seed=42):
    #model_rec / model_val: one model or a list of models (one per head, the risk is the mean)
    #model_rec was trained on ds_periodo and makes the recommendations.
    #model_val was trained on ANOTHER period: it never saw these moments, and checks the recommendation.
    #If the recommendation is real, model_val also sees the risk going down,
    #and doing the opposite should NOT lower the risk
    rng = np.random.default_rng(seed)
    limites = create_limits(ds_periodo, variaveis_controle)
    suporte = create_support(ds_periodo, variaveis_controle)

    #Moments with high predicted risk (top 10%) and all tags with value
    ds_periodo = ds_periodo[ds_periodo[variaveis_controle].notna().all(axis=1)].reset_index(drop=True)
    risco = predict_risk(model_rec, ds_periodo[features])
    indices = np.where(risco >= np.quantile(risco, 0.9))[0]
    indices = rng.choice(indices, size=min(n_momentos, len(indices)), replace=False)

    linhas = []
    for i in indices:
        linha = ds_periodo.iloc[i]
        recomendacao = recommend_actions(model_rec, linha, features, variaveis_controle, limites, suporte)
        if len(recomendacao['mudancas']) == 0:
            continue
        original = linha[features].astype(float)
        recomendada = recomendacao['linha_recomendada'][features].astype(float)
        contraria = opposite_change(original, recomendada, variaveis_controle)
        linhas.append({
            'Data': linha['Data'],
            'risco_rec_antes': recomendacao['risco_atual'],
            'risco_rec_depois': recomendacao['risco_recomendado'],
            'risco_val_antes': predict_risk(model_val, original.to_frame().T)[0],
            'risco_val_depois': predict_risk(model_val, recomendada.to_frame().T)[0],
            'risco_val_contrario': predict_risk(model_val, contraria.to_frame().T)[0],
            'mudancas': recomendacao['mudancas']
        })
    tabela = pd.DataFrame(linhas)
    if len(tabela) == 0:
        print('Nenhum momento recebeu recomendacao')
        return tabela, pd.Series(dtype=int)

    reducao_rec = tabela['risco_rec_antes'] - tabela['risco_rec_depois']
    reducao_val = tabela['risco_val_antes'] - tabela['risco_val_depois']
    reducao_contraria = tabela['risco_val_antes'] - tabela['risco_val_contrario']
    print('Momentos avaliados:', len(tabela))
    print('Reducao de risco segundo o modelo que recomendou:', round(reducao_rec.mean(), 3))
    print('Reducao de risco segundo o modelo do outro periodo:', round(reducao_val.mean(), 3))
    print('Reducao fazendo o contrario (outro periodo):', round(reducao_contraria.mean(), 3))
    print('Outro periodo concorda que o risco cai:', round(100*(reducao_val > 0).mean()), '%')
    print('Recomendacao melhor que fazer o contrario:', round(100*(reducao_val > reducao_contraria).mean()), '%')

    #How often each direction was recommended
    contagem = {}
    for mudancas in tabela['mudancas']:
        for mudanca in mudancas:
            chave = mudanca['variavel'] + ' ' + mudanca['direcao']
            contagem[chave] = contagem.get(chave, 0) + 1
    direcoes = pd.Series(contagem).sort_values(ascending=False)
    print('Direcoes recomendadas:')
    print(direcoes.to_string())
    print()
    return tabela, direcoes


def recommend_now(ds_historico, ds_agora, modelos_cabeca, features, variaveis_controle, passo_maximo=0.5):
    #Recommendation for the LAST instant of ds_agora (table made by features.create_model_features)
    #Risk = mean probability of plume of the heads that use a model (the tags enter only in them:
    #the persistence heads use only the camera score now, a change in the tags does not move them)
    #Allowed values come from the history (ds_historico): range and combinations the plant already operated
    limites = create_limits(ds_historico, variaveis_controle)
    suporte = create_support(ds_historico, variaveis_controle)
    linha = ds_agora.sort_values('Data').iloc[-1]
    lista_modelos = []
    for minutos in modelos_cabeca:
        lista_modelos.append(modelos_cabeca[minutos])
    recomendacao = recommend_actions(lista_modelos, linha, features, variaveis_controle, limites, suporte, passo_maximo)
    #Risk of each head, before and after the recommendation
    antes = linha[features].astype(float).to_frame().T
    depois = recomendacao['linha_recomendada'][features].astype(float).to_frame().T
    risco_cabeca = {}
    for minutos in modelos_cabeca:
        risco_antes = round(predict_risk(modelos_cabeca[minutos], antes)[0], 3)
        risco_depois = round(predict_risk(modelos_cabeca[minutos], depois)[0], 3)
        risco_cabeca[minutos] = (risco_antes, risco_depois)
    recomendacao['Data'] = linha['Data']
    recomendacao['risco_por_cabeca'] = risco_cabeca
    recomendacao['validada'] = RECOMENDACAO_VALIDADA
    return recomendacao


def main():
    #Step 5 of the pipeline: changes in the controllable tags that lower the risk of plume, for the
    #LAST instant of the interim. History (limits of operation) = processed table of the training
    ds_historico = load_processed()
    ds_agora = create_model_features(load_interim())
    modelos_cabeca = load_head_models(PREVISOR_POR_CABECA)
    recomendacao = recommend_now(ds_historico, ds_agora, modelos_cabeca, MODELO_ATUAL, VARIAVEIS_CONTROLE)
    print('Risco = media da probabilidade de pluma nas cabecas', list(modelos_cabeca.keys()), 'min')
    print_recommendation(recomendacao['Data'], recomendacao)
    return recomendacao


if __name__ == "__main__":
    main()
