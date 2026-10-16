import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import TimeSeriesSplit

from ml_atmospheric_emission.features import head_features
from ml_atmospheric_emission.modeling.train import create_model, train_heads_cv


def average_precision(y_true, score):
    #Same result as sklearn average_precision_score, but much faster (the bootstrap calls it many times)
    y_true = np.asarray(y_true)
    score = np.asarray(score)
    ordem = np.argsort(-score, kind='mergesort')
    score = score[ordem]
    y_true = y_true[ordem]
    #Thresholds: the last position of each distinct score value (ties are one threshold)
    ultimos = np.r_[np.where(np.diff(score) != 0)[0], len(score) - 1]
    verdadeiros = np.cumsum(y_true)[ultimos]
    precisao = verdadeiros / (ultimos + 1)
    recall = verdadeiros / verdadeiros[-1]
    recall_anterior = np.r_[0, recall[:-1]]
    return np.sum((recall - recall_anterior) * precisao)


def target_name(deslocamento_):
    #Name of the target column for a shift (each shift is 5 minutes)
    if deslocamento_ == 0:
        return 'y'
    return f'y_{deslocamento_*5}'


def split_holdout(ds, test_size=0.2):
    #Separates the hold out ONCE, by time: the last 20% of the period is never used in the analysis
    ds = ds.sort_values('Data').reset_index(drop=True)
    corte = ds['Data'].iloc[int(len(ds) * (1 - test_size))]
    ds_treino = ds[ds['Data'] < corte].reset_index(drop=True)
    ds_holdout = ds[ds['Data'] >= corte].reset_index(drop=True)
    #Purge: future targets that would look inside the hold out become NaN
    for coluna in ds_treino.columns:
        if coluna.startswith('y_'):
            minutos = int(coluna.split('_')[1])
            ds_treino.loc[ds_treino['Data'] + pd.Timedelta(minutes=minutos) >= corte, coluna] = np.nan
    print('Treino ate:', corte, '| linhas treino:', len(ds_treino), '| linhas hold out:', len(ds_holdout))
    return ds_treino, ds_holdout


def ap_cv(ds, features, target, gap, n_splits=5, limiar=0.5):
    #Average precision of each fold of the time series cross validation
    #ds must be ONLY the train period (use split_holdout before)
    ds = ds.sort_values('Data').reset_index(drop=True)
    ds = ds[ds[target].notna()].reset_index(drop=True)
    X = ds[features]
    y = (ds[target] >= limiar).astype(int)

    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    lista_ap = []
    lista_prevalencia = []
    for idx_train, idx_val in tscv.split(X):
        #Skipping folds with only one class
        if y.iloc[idx_train].nunique() < 2 or y.iloc[idx_val].sum() == 0:
            continue
        model = create_model()
        model.fit(X.iloc[idx_train], y.iloc[idx_train])
        prob_val = model.predict_proba(X.iloc[idx_val])[:, 1]
        lista_ap.append(average_precision_score(y.iloc[idx_val], prob_val))
        lista_prevalencia.append(y.iloc[idx_val].mean())
    return np.array(lista_ap), np.array(lista_prevalencia)


def signal_by_horizon(ds, tags, calendario, qtd_deslocamento):
    #For each horizon: model with tags + calendar against model with calendar only
    linhas = []
    for deslocamento_ in range(0, qtd_deslocamento + 1):
        target = target_name(deslocamento_)
        ap_calendario, prevalencia = ap_cv(ds, calendario, target, gap=deslocamento_)
        ap_tags, prevalencia = ap_cv(ds, tags + calendario, target, gap=deslocamento_)
        linhas.append({
            'minutos': deslocamento_*5,
            'prevalencia': round(np.mean(prevalencia), 3),
            'ap_calendario': round(np.mean(ap_calendario), 3),
            'ap_tags': round(np.mean(ap_tags), 3),
            'ganho_tags': round(np.mean(ap_tags) - np.mean(ap_calendario), 3),
            'folds_tags_vence': int(np.sum(ap_tags > ap_calendario))
        })
    tabela = pd.DataFrame(linhas)
    print(tabela.to_string(index=False))
    return tabela


def shuffle_days(ds, tags, rng):
    #Each complete day receives the tags of another day, at the same time of day:
    #keeps the daily pattern of the tags, but breaks their link with the smog of that day
    dx = ds.copy()
    dias = dx['Data'].dt.normalize()
    contagem = dias.value_counts()
    dias_completos = contagem[contagem >= 280].index.values
    mapa = dict(zip(dias_completos, rng.permutation(dias_completos)))
    #Incomplete days keep their own tags
    dia_origem = dias.map(mapa).fillna(dias)
    origem = dia_origem + (dx['Data'] - dias)
    tags_tempo = dx.set_index('Data')[tags]
    dx[tags] = tags_tempo.reindex(origem).values
    return dx


def gain_tags(ds, tags, calendario, lista_deslocamentos, lista_ap_calendario):
    #Mean gain of AP when the tags are added to the calendar, over the horizons
    lista_ganho = []
    cont = 0
    for deslocamento_ in lista_deslocamentos:
        ap_tags, prevalencia = ap_cv(ds, tags + calendario, target_name(deslocamento_), gap=deslocamento_)
        lista_ganho.append(np.mean(ap_tags) - lista_ap_calendario[cont])
        cont = cont+1
    return np.mean(lista_ganho)


def permutation_test_tags(ds, tags, calendario, lista_deslocamentos, n_perm=100, seed=42):
    #Is the gain of the tags real? The days are shuffled (shuffle_days) and the models retrained.
    #The same shuffle is used for all horizons, so the result is ONE test for all of them
    #ds must be ONLY the train period (use split_holdout before)
    rng = np.random.default_rng(seed)
    ds = ds.sort_values('Data').reset_index(drop=True)

    #The calendar model does not change when the tags are shuffled: computed only once
    lista_ap_calendario = []
    for deslocamento_ in lista_deslocamentos:
        ap_calendario, prevalencia = ap_cv(ds, calendario, target_name(deslocamento_), gap=deslocamento_)
        lista_ap_calendario.append(np.mean(ap_calendario))
    ganho_real = gain_tags(ds, tags, calendario, lista_deslocamentos, lista_ap_calendario)

    lista_ganho_perm = []
    for p in range(n_perm):
        dx = shuffle_days(ds, tags, rng)
        lista_ganho_perm.append(gain_tags(dx, tags, calendario, lista_deslocamentos, lista_ap_calendario))

    cont = 0
    for ganho_perm in lista_ganho_perm:
        if ganho_perm >= ganho_real:
            cont = cont+1
    p_valor = (1 + cont) / (1 + n_perm)
    resultado = {
        'minutos': [deslocamento_*5 for deslocamento_ in lista_deslocamentos],
        'ganho_real': round(ganho_real, 4),
        'ganho_acaso_medio': round(np.mean(lista_ganho_perm), 4),
        'ganho_acaso_p95': round(np.percentile(lista_ganho_perm, 95), 4),
        'p_valor': round(p_valor, 3)
    }
    print(resultado)
    return resultado


def lead_lag(ds, tags, lista_minutos):
    #Spearman correlation between each tag now and the smog score some minutes later
    #minutes > 0: the tag comes BEFORE the smog (can be a warning or a cause)
    #minutes < 0: the tag comes AFTER the smog (probably a consequence)
    #The hour of day pattern is removed first (the camera only sees the plume during the day)
    ds = ds.sort_values('Data').reset_index(drop=True)
    ds = ds.copy()
    for coluna in tags + ['y']:
        ds[coluna + '_res'] = ds[coluna] - ds.groupby('Hora')[coluna].transform('mean')
    y_tempo = ds.set_index('Data')['y_res']
    tabela = pd.DataFrame(index=tags)
    for minutos in lista_minutos:
        y_futuro = ds['Data'].map(y_tempo.shift(-minutos, freq='min'))
        for tag in tags:
            mascara = y_futuro.notna() & ds[tag + '_res'].notna()
            tabela.loc[tag, minutos] = ds.loc[mascara, tag + '_res'].corr(y_futuro[mascara], method='spearman')
    tabela = tabela.astype(float).round(3)
    print(tabela.to_string())
    return tabela


def lead_lag_null(ds, tags, n_perm=200, tamanho_bloco=288, seed=42):
    #Approximate correlation we would see by chance: shift the smog by 1 day or more
    rng = np.random.default_rng(seed)
    ds = ds.sort_values('Data').reset_index(drop=True)
    ds = ds.copy()
    for coluna in tags + ['y']:
        ds[coluna + '_res'] = ds[coluna] - ds.groupby('Hora')[coluna].transform('mean')
    y = ds['y_res'].values
    n = len(ds)
    limites = {}
    for tag in tags:
        lista_corr = []
        for p in range(n_perm):
            deslocamento = rng.integers(tamanho_bloco, n - tamanho_bloco)
            y_perm = pd.Series(np.roll(y, deslocamento))
            lista_corr.append(abs(ds[tag + '_res'].corr(y_perm, method='spearman')))
        limites[tag] = round(np.percentile(lista_corr, 95), 3)
    print('Correlacao maxima esperada por acaso (95%):', limites)
    return limites


def oof_predictions(ds, target, gap, modelos, n_splits=5, limiar=0.5, class_weight=None):
    #Out-of-fold predictions of the time series CV: each fold is predicted by models that only saw the past
    #Persistence: the forecast for t+k is the smog score now (column 'y'), no training needed
    #ds must be ONLY the train period (use split_holdout before)
    ds = ds.sort_values('Data').reset_index(drop=True)
    ds = ds[ds[target].notna()].reset_index(drop=True)
    y = (ds[target] >= limiar).astype(int)
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    lista = []
    fold = 0
    for idx_train, idx_val in tscv.split(ds):
        #Skipping folds with only one class
        if y.iloc[idx_train].nunique() < 2 or y.iloc[idx_val].sum() == 0:
            continue
        dx = pd.DataFrame({
            'fold': fold,
            'Data': ds['Data'].iloc[idx_val].values,
            'y_true': y.iloc[idx_val].values,
            'persistencia': ds['y'].iloc[idx_val].values
        })
        for nome in modelos:
            model = create_model(class_weight=class_weight)
            model.fit(ds[modelos[nome]].iloc[idx_train], y.iloc[idx_train])
            dx[nome] = model.predict_proba(ds[modelos[nome]].iloc[idx_val])[:, 1]
        lista.append(dx)
        fold = fold+1
    return pd.concat(lista).reset_index(drop=True)


def ap_by_fold(ds_oof, coluna):
    #AP of each fold, then the mean: each fold has its own model, so the folds are not mixed
    lista_ap = []
    for fold in ds_oof['fold'].unique():
        dx = ds_oof[ds_oof['fold'] == fold]
        lista_ap.append(average_precision(dx['y_true'].values, dx[coluna].values))
    return np.mean(lista_ap)


def bootstrap_days(ds_oof, colunas, n_boot=1000, seed=42):
    #Paired bootstrap by days: inside each fold, draws whole days with replacement
    #and recomputes the mean AP by fold of every column (all columns use the same days)
    #Whole days keep the autocorrelation inside the day, and no normal distribution is assumed
    rng = np.random.default_rng(seed)
    dias = ds_oof['Data'].dt.normalize()
    indices_fold = {}
    for fold in ds_oof['fold'].unique():
        indices_dia = {}
        for dia in dias[ds_oof['fold'] == fold].unique():
            indices_dia[dia] = np.where((ds_oof['fold'] == fold) & (dias == dia))[0]
        indices_fold[fold] = indices_dia
    y_true = ds_oof['y_true'].values
    linhas = []
    for b in range(n_boot):
        soma = {}
        for coluna in colunas:
            soma[coluna] = 0
        valido = True
        for fold in indices_fold:
            lista_dias = list(indices_fold[fold].keys())
            sorteados = rng.choice(len(lista_dias), size=len(lista_dias), replace=True)
            indices = np.concatenate([indices_fold[fold][lista_dias[i]] for i in sorteados])
            if y_true[indices].sum() == 0:
                valido = False
                break
            for coluna in colunas:
                soma[coluna] = soma[coluna] + average_precision(y_true[indices], ds_oof[coluna].values[indices])
        if not valido:
            continue
        linha = {}
        for coluna in colunas:
            linha[coluna] = soma[coluna] / len(indices_fold)
        linhas.append(linha)
    return pd.DataFrame(linhas)


def holm(p_values):
    #Holm correction: many heads tested at once, so each p-value is adjusted
    p_values = np.array(p_values)
    m = len(p_values)
    ordem = np.argsort(p_values)
    p_ajustado = np.zeros(m)
    maximo = 0
    cont = 0
    for i in ordem:
        valor = min(1, (m - cont) * p_values[i])
        maximo = max(maximo, valor)
        p_ajustado[i] = maximo
        cont = cont+1
    return p_ajustado


def compare_horizons(ds, lista_deslocamentos, modelos, minutos_teste, n_boot=1000):
    #For each horizon: mean AP by fold of the persistence and of each model (CV, train period only)
    #For the heads in minutos_teste: bootstrap test "model is better than the reference"
    linhas = []
    colunas = ['persistencia'] + list(modelos.keys())
    for deslocamento_ in lista_deslocamentos:
        ds_oof = oof_predictions(ds, target_name(deslocamento_), deslocamento_, modelos)
        linha = {'minutos': deslocamento_*5}
        for coluna in colunas:
            linha['ap_' + coluna] = ap_by_fold(ds_oof, coluna)
        if deslocamento_*5 in minutos_teste:
            ds_boot = bootstrap_days(ds_oof, colunas, n_boot)
            for nome in modelos:
                for referencia in ['persistencia', 'calendario']:
                    if nome == referencia:
                        continue
                    diferenca = ds_boot[nome] - ds_boot[referencia]
                    linha['dif_' + nome + '_vs_' + referencia] = linha['ap_' + nome] - linha['ap_' + referencia]
                    linha['ic_inf_' + nome + '_vs_' + referencia] = np.percentile(diferenca, 2.5)
                    linha['ic_sup_' + nome + '_vs_' + referencia] = np.percentile(diferenca, 97.5)
                    linha['p_' + nome + '_vs_' + referencia] = (1 + np.sum(diferenca <= 0)) / (1 + len(diferenca))
        linhas.append(linha)
        print('minutos:', deslocamento_*5, '| AP persistencia:', round(linha['ap_persistencia'], 3),
              '| AP modelos:', [round(linha['ap_' + nome], 3) for nome in modelos])
    return pd.DataFrame(linhas)


def choose_horizons(tabela, modelo, referencias, alpha=0.05):
    #A head is chosen only if the model beats ALL the references
    #Holm correction over the tested heads (the rows that have a p-value)
    testadas = tabela[tabela['p_' + modelo + '_vs_' + referencias[0]].notna()].reset_index(drop=True)
    vence_todas = np.ones(len(testadas), dtype=bool)
    for referencia in referencias:
        p_holm = holm(testadas['p_' + modelo + '_vs_' + referencia].values)
        print('  p de Holm', modelo, 'vs', referencia, ':', dict(zip(testadas['minutos'].astype(int), np.round(p_holm, 3))))
        vence_todas = vence_todas & (p_holm < alpha)
    escolhidos = [int(minutos) for minutos in testadas['minutos'][vence_todas]]
    print('Modelo', modelo, '| vence', referencias, '| cabecas:', escolhidos)
    return escolhidos


def choose_forecaster(tabela, modelo, alpha=0.05):
    #Indicator rule for each head: use the model only where it beats the persistence
    #(bootstrap by days, Holm over the tested heads). If not, use the persistence (camera score now)
    testadas = tabela[tabela['p_' + modelo + '_vs_persistencia'].notna()].reset_index(drop=True)
    p_holm = holm(testadas['p_' + modelo + '_vs_persistencia'].values)
    previsores = {}
    for i in range(len(testadas)):
        minutos = int(testadas['minutos'].iloc[i])
        if p_holm[i] < alpha:
            previsores[minutos] = modelo
        else:
            previsores[minutos] = 'persistencia'
    print('p de Holm', modelo, 'vs persistencia:', dict(zip(testadas['minutos'].astype(int), np.round(p_holm, 3))))
    print('Previsor por cabeca:', previsores)
    return previsores


def holdout_ap(ds_treino, ds_holdout, lista_minutos, modelos, n_boot=1000, limiar=0.5):
    #Final check, used only once at the end: train on the whole train period, AP on the hold out,
    #and the bootstrap by days of (model - persistence) on the hold out
    linhas = []
    for minutos in lista_minutos:
        target = target_name(minutos // 5)
        treino = ds_treino[ds_treino[target].notna()]
        teste = ds_holdout[ds_holdout[target].notna()].reset_index(drop=True)
        y_treino = (treino[target] >= limiar).astype(int)
        y_teste = (teste[target] >= limiar).astype(int)
        ds_teste = pd.DataFrame({'fold': 0, 'Data': teste['Data'], 'y_true': y_teste, 'persistencia': teste['y']})
        for nome in modelos:
            model = create_model(class_weight=None)
            model.fit(treino[modelos[nome]], y_treino)
            ds_teste[nome] = model.predict_proba(teste[modelos[nome]])[:, 1]
        ds_boot = bootstrap_days(ds_teste, ['persistencia'] + list(modelos.keys()), n_boot)
        linha = {'minutos': minutos, 'prevalencia_teste': round(y_teste.mean(), 3)}
        linha['ap_teste_persistencia'] = round(average_precision_score(y_teste, ds_teste['persistencia']), 3)
        for nome in modelos:
            linha['ap_teste_' + nome] = round(average_precision_score(y_teste, ds_teste[nome]), 3)
            diferenca = ds_boot[nome] - ds_boot['persistencia']
            linha['ic_inf_' + nome + '_vs_persistencia'] = round(np.percentile(diferenca, 2.5), 3)
            linha['ic_sup_' + nome + '_vs_persistencia'] = round(np.percentile(diferenca, 97.5), 3)
        linhas.append(linha)
    tabela = pd.DataFrame(linhas)
    print(tabela.to_string(index=False))
    return tabela


def save_horizon_report(tabela_cv, tabela_teste, previsores, info, pasta):
    #Saves the tables (csv) and a short markdown report: WHY each head uses each forecaster
    tabela_cv.to_csv(pasta / 'escolha_horizontes_cv.csv', index=False)
    tabela_teste.to_csv(pasta / 'escolha_horizontes_teste.csv', index=False)
    #Only the tested heads, only the AP, difference and p-value columns
    testadas = tabela_cv[tabela_cv['minutos'].isin(tabela_teste['minutos'])]
    colunas = []
    for coluna in testadas.columns:
        if coluna == 'minutos' or coluna.startswith('ap_') or coluna.startswith('dif_') or coluna.startswith('p_'):
            colunas.append(coluna)
    texto = '# Escolha dos horizontes (cabecas) do modelo\n\n'
    texto = texto + 'Gerado por notebooks/3.0-jm-escolha-horizontes.ipynb em ' + str(pd.Timestamp.now().round('s')) + '\n\n'
    for chave in info:
        texto = texto + '- ' + chave + ': ' + str(info[chave]) + '\n'
    texto = texto + '\n## Regra\n\n'
    texto = texto + '- Metrica: AP (average precision) por fold do CV temporal (so periodo de treino), media dos folds.\n'
    texto = texto + '- Referencias: persistencia (score da camera agora) e modelo so com o calendario.\n'
    texto = texto + '- Teste: bootstrap pareado por dias dentro de cada fold, p-valor unilateral, Holm nas cabecas testadas.\n'
    texto = texto + '- Indicadora: em cada cabeca, usa o modelo so se ele vence a persistencia (p de Holm < 0,05); '
    texto = texto + 'se nao, usa a persistencia.\n'
    texto = texto + '\n## Resultado: previsor por cabeca (PREVISOR_POR_CABECA)\n\n'
    for minutos in previsores:
        texto = texto + '- ' + str(minutos) + ' min: ' + previsores[minutos] + '\n'
    texto = texto + '\n## CV temporal (treino)\n\n```\n' + testadas[colunas].round(3).to_string(index=False) + '\n```\n'
    texto = texto + '\n## Hold out (teste, usado uma vez so)\n\n```\n' + tabela_teste.to_string(index=False) + '\n```\n'
    texto = texto + '\n![Horizontes](figures/horizontes_vs_persistencia.png)\n'
    caminho = pasta / 'escolha_horizontes.md'
    with open(caminho, 'w', encoding='utf-8') as arquivo:
        arquivo.write(texto)
    print('Report salvo em', caminho)


def factorial_configurations(base, grupos):
    #All the combinations of the groups (each group in or out): 2^len(grupos) configurations
    #Returns {name: list of columns}. 'atual' is the base without any group
    nomes = list(grupos.keys())
    configuracoes = {}
    for combinacao in itertools.product([0, 1], repeat=len(nomes)):
        features = list(base)
        partes = []
        for i in range(len(nomes)):
            if combinacao[i] == 1:
                features = features + grupos[nomes[i]]
                partes.append(nomes[i])
        if len(partes) == 0:
            nome = 'atual'
        else:
            nome = 'atual+' + '+'.join(partes)
        configuracoes[nome] = features
    print('Configuracoes:', len(configuracoes), '= 2 ^', len(nomes))
    return configuracoes


def factorial_blocks(base, blocos):
    #Factorial by blocks: ALL the combinations inside each block (2^n of each block), on top of the base
    #blocos = {block name: {group name: columns}}. 'atual' (the base alone) is in every block, so it appears once
    #Combinations of groups of DIFFERENT blocks are not tested (it would be 2^(all the groups))
    configuracoes = {}
    for nome_bloco in blocos:
        print('Bloco', nome_bloco, '| grupos:', list(blocos[nome_bloco].keys()))
        configuracoes_bloco = factorial_configurations(base, blocos[nome_bloco])
        for nome in configuracoes_bloco:
            configuracoes[nome] = configuracoes_bloco[nome]
    print('Total de configuracoes (a atual uma vez so):', len(configuracoes))
    return configuracoes


def factorial_cv(ds, configuracoes, cabecas):
    #Runs train_heads_cv for every configuration and joins the predictions of each head in one table
    #(all configurations have the same rows and folds, so the comparison is paired)
    #Returns {minutos: table with fold, Data, y_true, persistencia and one column per configuration}
    tabelas = {}
    cont = 1
    for nome in configuracoes:
        resultados = train_heads_cv(ds, configuracoes[nome], cabecas)
        for minutos in cabecas:
            if minutos not in tabelas:
                tabelas[minutos] = resultados[minutos][['fold', 'Data', 'y_true', 'persistencia']].copy()
            tabelas[minutos][nome] = resultados[minutos]['modelo'].values
        print(cont, '/', len(configuracoes), nome)
        cont = cont+1
    return tabelas


def reality_check(ds_oof, referencia, candidatos, n_boot=1000, seed=42):
    #Is the BEST candidate better than the reference? (White's Reality Check, with the bootstrap by days)
    #Testing many configurations, one of them looks better just by chance:
    #the test compares the best real gain with the best gain that appears by chance in the bootstrap
    ds_boot = bootstrap_days(ds_oof, [referencia] + candidatos, n_boot, seed)
    ganho_real = {}
    for nome in candidatos:
        ganho_real[nome] = ap_by_fold(ds_oof, nome) - ap_by_fold(ds_oof, referencia)
    melhor = max(ganho_real, key=ganho_real.get)
    #Bootstrap gains centered on the real gains: how the best gain looks when there is no real improvement
    maximo_acaso = np.full(len(ds_boot), -np.inf)
    for nome in candidatos:
        ganho_boot = ds_boot[nome].values - ds_boot[referencia].values
        maximo_acaso = np.maximum(maximo_acaso, ganho_boot - ganho_real[nome])
    p_valor = (1 + np.sum(maximo_acaso >= ganho_real[melhor])) / (1 + len(ds_boot))
    diferenca = ds_boot[melhor].values - ds_boot[referencia].values
    resultado = {
        'ap_referencia': ap_by_fold(ds_oof, referencia),
        'melhor': melhor,
        'ap_melhor': ap_by_fold(ds_oof, melhor),
        'ganho': ganho_real[melhor],
        'ic_inf': np.percentile(diferenca, 2.5),
        'ic_sup': np.percentile(diferenca, 97.5),
        'p_valor': p_valor
    }
    return resultado


def find_configuration(configuracoes, features):
    #Name of the configuration that has exactly these columns (None if no configuration has them)
    for nome in configuracoes:
        if set(configuracoes[nome]) == set(features):
            return nome
    return None


def choose_new_forecasters(tabelas, configuracoes, previsores_atuais, modelos, n_boot=1000, alpha=0.05):
    #For each head: the reference is the CURRENT forecaster of the head:
    #'persistencia', or the configuration with the same columns as modelos[current forecaster]
    #Reality Check of the best configuration against it, then Holm over the heads.
    #The forecaster of the head changes only if the improvement is significant
    linhas = []
    for minutos in tabelas:
        if previsores_atuais[minutos] == 'persistencia':
            referencia = 'persistencia'
            candidatos = list(configuracoes.keys())
        else:
            referencia = find_configuration(configuracoes, modelos[previsores_atuais[minutos]])
            if referencia is None:
                print(minutos, 'min: o previsor atual', previsores_atuais[minutos], 'nao esta entre as configuracoes testadas')
                continue
            candidatos = [nome for nome in configuracoes if nome != referencia]
        resultado = reality_check(tabelas[minutos], referencia, candidatos, n_boot)
        resultado['minutos'] = minutos
        resultado['previsor_atual'] = previsores_atuais[minutos]
        resultado['referencia'] = referencia
        linhas.append(resultado)
        print(minutos, 'min | referencia:', referencia, '| melhor:', resultado['melhor'],
              '| ganho:', round(resultado['ganho'], 3), '| p:', round(resultado['p_valor'], 3))
    tabela = pd.DataFrame(linhas)
    tabela['p_holm'] = holm(tabela['p_valor'].values)
    tabela['melhoria'] = tabela['p_holm'] < alpha
    novo_previsor = []
    for i in range(len(tabela)):
        if tabela['melhoria'].iloc[i]:
            #If the winner has the same columns as a model that already exists in modelos, use that name
            nome = tabela['melhor'].iloc[i]
            for nome_modelo in modelos:
                if set(modelos[nome_modelo]) == set(configuracoes[nome]):
                    nome = nome_modelo
            novo_previsor.append(nome)
        else:
            novo_previsor.append(tabela['previsor_atual'].iloc[i])
    tabela['novo_previsor'] = novo_previsor
    colunas = ['minutos', 'previsor_atual', 'referencia', 'ap_referencia', 'melhor', 'ap_melhor',
               'ganho', 'ic_inf', 'ic_sup', 'p_valor', 'p_holm', 'melhoria', 'novo_previsor']
    return tabela[colunas]


def main_effects(tabelas, grupos):
    #Factorial analysis: for each group and head, mean AP of the configurations WITH the group
    #minus the mean AP of the configurations WITHOUT it (the average gain of turning the group on)
    #Only the configurations made with the groups of grupos (in the factorial by blocks: the block of the group)
    efeitos = pd.DataFrame(index=list(grupos.keys()))
    for minutos in tabelas:
        configuracoes = []
        for coluna in tabelas[minutos].columns:
            if not coluna.startswith('atual'):
                continue
            dentro = True
            for parte in coluna.split('+')[1:]:
                if parte not in grupos:
                    dentro = False
            if dentro:
                configuracoes.append(coluna)
        ap = {}
        for nome in configuracoes:
            ap[nome] = ap_by_fold(tabelas[minutos], nome)
        for grupo in grupos:
            com = [ap[nome] for nome in configuracoes if grupo in nome.split('+')]
            sem = [ap[nome] for nome in configuracoes if grupo not in nome.split('+')]
            efeitos.loc[grupo, minutos] = np.mean(com) - np.mean(sem)
    return efeitos.astype(float)


def holdout_compare(ds_treino, ds_holdout, minutos, features_novas, features_referencia, n_boot=1000, limiar=0.5):
    #Final check of a change (used only once): train on the whole train period, compare on the hold out
    #features_referencia = None means the reference is the persistence
    target = 'y_' + str(minutos)
    treino = ds_treino[ds_treino[target].notna()]
    teste = ds_holdout[ds_holdout[target].notna()].reset_index(drop=True)
    y_treino = (treino[target] >= limiar).astype(int)
    y_teste = (teste[target] >= limiar).astype(int)
    if y_teste.sum() == 0:
        print(minutos, 'min: o hold out nao tem pluma nesta cabeca, nada para comparar')
        return {'minutos': minutos, 'ap_teste_novo': np.nan, 'ap_teste_referencia': np.nan, 'ic_inf': np.nan, 'ic_sup': np.nan}
    ds_teste = pd.DataFrame({'fold': 0, 'Data': teste['Data'], 'y_true': y_teste})
    colunas_novas = head_features(features_novas, minutos)
    model = create_model(class_weight=None)
    model.fit(treino[colunas_novas], y_treino)
    ds_teste['novo'] = model.predict_proba(teste[colunas_novas])[:, 1]
    if features_referencia is None:
        ds_teste['referencia'] = teste['y'].values
    else:
        colunas_referencia = head_features(features_referencia, minutos)
        model = create_model(class_weight=None)
        model.fit(treino[colunas_referencia], y_treino)
        ds_teste['referencia'] = model.predict_proba(teste[colunas_referencia])[:, 1]
    ds_boot = bootstrap_days(ds_teste, ['novo', 'referencia'], n_boot)
    diferenca = ds_boot['novo'] - ds_boot['referencia']
    resultado = {
        'minutos': minutos,
        'ap_teste_novo': average_precision(ds_teste['y_true'].values, ds_teste['novo'].values),
        'ap_teste_referencia': average_precision(ds_teste['y_true'].values, ds_teste['referencia'].values),
        'ic_inf': np.percentile(diferenca, 2.5),
        'ic_sup': np.percentile(diferenca, 97.5)
    }
    return resultado


def compare_runs(resultados_referencia, resultados_candidato, n_boot=1000, alpha=0.05):
    #Paired test of two results of train_heads_cv (same rows and folds) in each head of the candidate:
    #is the candidate better than the reference? Bootstrap by days of (candidate - reference).
    #p-value = fraction of the bootstrap where the candidate is NOT better (same rule of the notebook 3.0,
    #it agrees with the 95% interval), then Holm over the heads. The AP of the persistence goes as context
    linhas = []
    for minutos in resultados_candidato:
        ds_oof = resultados_referencia[minutos].rename(columns={'modelo': 'referencia'})
        if not (ds_oof['Data'].values == resultados_candidato[minutos]['Data'].values).all():
            raise ValueError(str(minutos) + ' min: as duas rodadas nao tem as mesmas linhas')
        ds_oof['candidato'] = resultados_candidato[minutos]['modelo'].values
        ds_boot = bootstrap_days(ds_oof, ['referencia', 'candidato'], n_boot)
        diferenca = ds_boot['candidato'] - ds_boot['referencia']
        linhas.append({
            'minutos': minutos,
            'ap_persistencia': ap_by_fold(ds_oof, 'persistencia'),
            'ap_referencia': ap_by_fold(ds_oof, 'referencia'),
            'ap_candidato': ap_by_fold(ds_oof, 'candidato'),
            'ganho': ap_by_fold(ds_oof, 'candidato') - ap_by_fold(ds_oof, 'referencia'),
            'ic_inf': np.percentile(diferenca, 2.5),
            'ic_sup': np.percentile(diferenca, 97.5),
            'p_valor': (1 + np.sum(diferenca <= 0)) / (1 + len(diferenca))
        })
    tabela = pd.DataFrame(linhas)
    tabela['p_holm'] = holm(tabela['p_valor'].values)
    tabela['melhoria'] = tabela['p_holm'] < alpha
    return tabela
