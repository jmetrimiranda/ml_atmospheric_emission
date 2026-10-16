import matplotlib.pyplot as plt
import pandas as pd


def plot_horizons(tabela, caminho, previsores=None):
    #Top: AP of each approach per horizon (mean by fold of the CV)
    #Bottom: gain of each model over the persistence, with the 95% interval (bootstrap by days), on the tested heads
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    series = [
        ('persistencia', 'persistencia', '#2a78d6', '--'),
        ('calendario', 'so calendario', '#eb6834', '--'),
        ('base', 'modelo base (tags + calendario)', '#1baf7a', '-'),
        ('base_estado', 'modelo + estado da camera', '#eda100', '-'),
    ]
    for coluna, nome, cor, estilo in series:
        ax1.plot(tabela['minutos'], tabela['ap_' + coluna], color=cor, linestyle=estilo, linewidth=2, marker='o', markersize=4, label=nome)
    ax1.set_title('AP no CV temporal por horizonte')
    ax1.set_ylabel('AP (media dos folds)')
    ax1.set_ylim(0, None)
    ax1.legend(frameon=False, loc='upper right')

    testadas = tabela[tabela['p_base_vs_persistencia'].notna()]
    deslocamento = 2
    for coluna, nome, cor in [('base', 'modelo base - persistencia', '#1baf7a'), ('base_estado', 'modelo + estado - persistencia', '#eda100')]:
        centro = testadas['dif_' + coluna + '_vs_persistencia']
        erro_baixo = centro - testadas['ic_inf_' + coluna + '_vs_persistencia']
        erro_alto = testadas['ic_sup_' + coluna + '_vs_persistencia'] - centro
        ax2.errorbar(testadas['minutos'] + deslocamento, centro, yerr=[erro_baixo, erro_alto], fmt='o', color=cor,
                     markersize=6, capsize=3, linewidth=1.5, label=nome)
        deslocamento = -2
    ax2.axhline(0, color='#555555', linewidth=1)
    #Indicator: dashed line where the forecaster changes from the persistence to the model
    if previsores is not None:
        cabecas_modelo = [minutos for minutos in previsores if previsores[minutos] != 'persistencia']
        cabecas_persistencia = [minutos for minutos in previsores if previsores[minutos] == 'persistencia']
        if len(cabecas_modelo) > 0 and len(cabecas_persistencia) > 0:
            troca = (max(cabecas_persistencia) + min(cabecas_modelo)) / 2
            for ax in [ax1, ax2]:
                ax.axvline(troca, color='#555555', linestyle=':', linewidth=1.5)
            ax1.text(troca, ax1.get_ylim()[1]*0.97, '  persistencia | modelo  ', ha='center', va='top', fontsize=9, color='#333333',
                     bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.8})
    ax2.set_title('Ganho de AP sobre a persistencia nas cabecas testadas (intervalo de 95%, bootstrap por dias)')
    ax2.set_xlabel('Minutos no futuro')
    ax2.set_ylabel('Diferenca de AP')
    ax2.legend(frameon=False, loc='lower right')

    for ax in [ax1, ax2]:
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(caminho, dpi=150, bbox_inches='tight')
    plt.show()


def plot_factorial(tabela_decisao, efeitos, caminho):
    #Top: for each head, gain of the best configuration over the current forecaster (95% interval)
    #filled point = significant improvement (Reality Check + Holm), empty point = not significant
    #Bottom: main effect of each group of features in each head (mean AP with the group - without it)
    from matplotlib.colors import LinearSegmentedColormap
    altura = 4 + 0.4*len(efeitos)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 4 + altura), gridspec_kw={'height_ratios': [4, altura]})
    posicoes = list(range(len(tabela_decisao)))
    for i in posicoes:
        linha = tabela_decisao.iloc[i]
        cor = '#2a78d6'
        ax1.plot([i, i], [linha['ic_inf'], linha['ic_sup']], color=cor, linewidth=1.5)
        if linha['melhoria']:
            ax1.plot(i, linha['ganho'], 'o', color=cor, markersize=9)
        else:
            ax1.plot(i, linha['ganho'], 'o', markerfacecolor='white', markeredgecolor=cor, markeredgewidth=2, markersize=9)
        #Name without the 'atual+' of the start (solar_atual also has 'atual' in the name)
        nome = linha['melhor']
        if nome.startswith('atual+'):
            nome = nome[len('atual+'):]
        nome = nome.replace('+', '\n')
        ax1.annotate(nome, (i, linha['ic_sup']), xytext=(0, 4), textcoords='offset points', ha='center', va='bottom',
                     fontsize=7, color='#333333')
    ax1.axhline(0, color='#555555', linewidth=1)
    rotulos = [str(m) + ' min\n(' + str(p) + ')' for m, p in zip(tabela_decisao['minutos'], tabela_decisao['previsor_atual'])]
    ax1.set_xticks(posicoes)
    ax1.set_xticklabels(rotulos, fontsize=8)
    ax1.set_ylabel('Ganho de AP sobre o previsor atual')
    ax1.set_title('Melhor configuracao de cada cabeca: ganho e intervalo de 95% do melhor (sem ajuste pela escolha entre muitas)\n'
                  'cheio = melhoria significativa (Reality Check + Holm), vazio = nao significativa', fontsize=10)
    ax1.grid(True, axis='y', alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    margem = (tabela_decisao['ic_sup'].max() - tabela_decisao['ic_inf'].min()) * 0.35
    ax1.set_ylim(tabela_decisao['ic_inf'].min() - 0.01, tabela_decisao['ic_sup'].max() + margem)

    #Diverging colors: red = the group lowers the AP, gray = no effect, blue = the group raises the AP
    cores = LinearSegmentedColormap.from_list('divergente', ['#e34948', '#f0efec', '#2a78d6'])
    limite = max(abs(efeitos.values.min()), abs(efeitos.values.max()))
    imagem = ax2.imshow(efeitos.values, cmap=cores, vmin=-limite, vmax=limite, aspect='auto')
    for i in range(efeitos.shape[0]):
        for j in range(efeitos.shape[1]):
            ax2.text(j, i, f'{efeitos.values[i, j]:+.3f}', ha='center', va='center', fontsize=8, color='#222222')
    ax2.set_xticks(range(efeitos.shape[1]))
    ax2.set_xticklabels([str(m) + ' min' for m in efeitos.columns], fontsize=8)
    ax2.set_yticks(range(efeitos.shape[0]))
    ax2.set_yticklabels(efeitos.index, fontsize=9)
    ax2.set_title('Efeito principal de cada grupo de features (AP com o grupo - sem o grupo)')
    fig.colorbar(imagem, ax=ax2, label='Diferenca de AP', shrink=0.8)
    fig.tight_layout()
    fig.savefig(caminho, dpi=150, bbox_inches='tight')
    plt.show()


def plot_columns(ds, paineis, inicio, dias, caminho, titulo):
    #Some days of the chosen columns, one panel per list of columns
    #Gray band = plume (camera score >= 0.5), to see each feature next to the target
    cores = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
    fim = pd.Timestamp(inicio) + pd.Timedelta(days=dias)
    dx = ds[(ds['Data'] >= pd.Timestamp(inicio)) & (ds['Data'] < fim)].sort_values('Data')
    fig, eixos = plt.subplots(len(paineis), 1, figsize=(11, 2.8*len(paineis)), sharex=True, squeeze=False)
    cont = 0
    for colunas in paineis:
        ax = eixos[cont, 0]
        ax.fill_between(dx['Data'], 0, 1, where=(dx['y'] >= 0.5).values, transform=ax.get_xaxis_transform(),
                        color='#bbbbbb', alpha=0.4, linewidth=0, label='pluma (score >= 0,5)')
        i = 0
        for coluna in colunas:
            ax.plot(dx['Data'], dx[coluna], color=cores[i % len(cores)], linewidth=1.6, label=coluna)
            i = i+1
        ax.legend(frameon=False, fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1))
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        cont = cont+1
    eixos[0, 0].set_title(titulo)
    fig.tight_layout()
    fig.savefig(caminho, dpi=150, bbox_inches='tight')
    plt.show()


def plot_model_features(tabela_persistencia, tabela_cadeia, caminho):
    #Top: AP of each head (mean by fold of the CV) of the persistence (forecaster), of the model without the
    #persistence as a feature, of the current model and of the chain (current model + predictions of the heads before)
    #Bottom: gain of each test with the 95% interval (filled = significant after Holm, empty = not significant)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1.plot(tabela_persistencia['minutos'], tabela_persistencia['ap_persistencia'], color='#2a78d6', linestyle='--',
             linewidth=2, marker='o', markersize=4, label='persistencia (score agora, como previsor)')
    ax1.plot(tabela_persistencia['minutos'], tabela_persistencia['ap_referencia'], color='#eb6834', linewidth=2,
             marker='o', markersize=4, label='modelo SEM a persistencia como feature')
    ax1.plot(tabela_persistencia['minutos'], tabela_persistencia['ap_candidato'], color='#1baf7a', linewidth=2,
             marker='o', markersize=4, label='MODELO_ATUAL (com a persistencia)')
    ax1.plot(tabela_cadeia['minutos'], tabela_cadeia['ap_candidato'], color='#eda100', linewidth=2,
             marker='o', markersize=4, label='MODELO_ATUAL + previsoes das cabecas anteriores')
    ax1.set_title('AP no CV temporal por cabeca')
    ax1.set_ylabel('AP (media dos folds)')
    ax1.set_ylim(0, None)
    ax1.legend(frameon=False, fontsize=8, loc='upper right')

    deslocamento = -4
    for tabela, nome, cor in [(tabela_persistencia, 'efeito da persistencia (com - sem)', '#1baf7a'),
                              (tabela_cadeia, 'efeito da cadeia (com previsoes anteriores - atual)', '#eda100')]:
        primeiro = True
        for i in range(len(tabela)):
            linha = tabela.iloc[i]
            x = linha['minutos'] + deslocamento
            ax2.plot([x, x], [linha['ic_inf'], linha['ic_sup']], color=cor, linewidth=1.5)
            rotulo = None
            if primeiro:
                rotulo = nome
                primeiro = False
            if linha['melhoria']:
                ax2.plot(x, linha['ganho'], 'o', color=cor, markersize=8, label=rotulo)
            else:
                ax2.plot(x, linha['ganho'], 'o', markerfacecolor='white', markeredgecolor=cor, markeredgewidth=2,
                         markersize=8, label=rotulo)
        deslocamento = 4
    ax2.axhline(0, color='#555555', linewidth=1)
    ax2.set_title('Ganho de AP (intervalo de 95%, bootstrap por dias; cheio = significativo com Holm)')
    ax2.set_xlabel('Minutos no futuro (cabeca)')
    ax2.set_ylabel('Diferenca de AP')
    ax2.legend(frameon=False, fontsize=8, loc='upper right')
    for ax in [ax1, ax2]:
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(caminho, dpi=150, bbox_inches='tight')
    plt.show()


def plot_algorithms(tabela_decisao, ap_treino, caminho):
    #Top: AP of the CV (validation) of each algorithm per head, and the persistence (dashed)
    #Middle: AP in the TRAIN rows minus AP of the CV: the bigger, the more the model memorized the train
    #Bottom: gain of the best algorithm over the current forecaster of the head (95% interval of the best, filled = significant)
    cores = {'lightgbm': '#1baf7a', 'xgboost': '#eb6834', 'gradient_boosting': '#eda100', 'random_forest': '#8a5cd0'}
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    posicoes = list(range(len(tabela_decisao)))
    ax1.plot(posicoes, tabela_decisao['ap_persistencia'], color='#2a78d6', linestyle='--', linewidth=2, marker='o',
             markersize=4, label='persistencia')
    for nome in cores:
        if 'ap_' + nome in tabela_decisao.columns:
            ax1.plot(posicoes, tabela_decisao['ap_' + nome], color=cores[nome], linewidth=2, marker='o', markersize=4, label=nome)
            ax2.plot(posicoes, ap_treino[nome].values - tabela_decisao['ap_' + nome].values, color=cores[nome], linewidth=2,
                     marker='o', markersize=4, label=nome)
    ax1.set_title('AP no CV temporal (validacao) por cabeca')
    ax1.set_ylabel('AP (media dos folds)')
    ax1.set_ylim(0, None)
    ax1.legend(frameon=False, fontsize=8, loc='upper right')
    ax2.set_title('Overfitting: AP no treino - AP na validacao (quanto maior, mais decorado)')
    ax2.set_ylabel('Diferenca de AP')
    ax2.set_ylim(0, None)
    ax2.legend(frameon=False, fontsize=8, loc='upper left')

    for i in posicoes:
        linha = tabela_decisao.iloc[i]
        cor = cores.get(linha['melhor'], '#555555')
        ax3.plot([i, i], [linha['ic_inf'], linha['ic_sup']], color=cor, linewidth=1.5)
        if linha['melhoria']:
            ax3.plot(i, linha['ganho'], 'o', color=cor, markersize=9)
        else:
            ax3.plot(i, linha['ganho'], 'o', markerfacecolor='white', markeredgecolor=cor, markeredgewidth=2, markersize=9)
        ax3.annotate(linha['melhor'], (i, linha['ic_sup']), xytext=(0, 4), textcoords='offset points', ha='center',
                     va='bottom', fontsize=7, color='#333333')
    ax3.axhline(0, color='#555555', linewidth=1)
    ax3.set_title('Melhor algoritmo contra o previsor atual (cheio = significativo com Reality Check + Holm)')
    ax3.set_ylabel('Ganho de AP')
    margem = (tabela_decisao['ic_sup'].max() - tabela_decisao['ic_inf'].min()) * 0.25
    ax3.set_ylim(tabela_decisao['ic_inf'].min() - 0.01, tabela_decisao['ic_sup'].max() + margem)
    rotulos = [str(m) + ' min\n(vs ' + r + ')' for m, r in zip(tabela_decisao['minutos'], tabela_decisao['referencia'])]
    ax3.set_xticks(posicoes)
    ax3.set_xticklabels(rotulos, fontsize=8)
    for ax in [ax1, ax2, ax3]:
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(caminho, dpi=150, bbox_inches='tight')
    plt.show()


def plot_persistence(tabela, caminho):
    #AP of the persistence (forecast for t + k = camera score now) in each head, and the prevalence of plume
    #(the AP of a random guess). When the two lines meet, the score now says nothing about t + k
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(tabela['minutos'], tabela['ap_persistencia'], color='#2a78d6', linewidth=2, marker='o', markersize=5,
            label='AP da persistencia (score agora como previsao)')
    ax.plot(tabela['minutos'], tabela['prevalencia'], color='#888888', linestyle='--', linewidth=1.5,
            label='prevalencia de pluma (AP de um chute)')
    for i in range(len(tabela)):
        ax.annotate(f"{tabela['ap_persistencia'].iloc[i]:.2f}", (tabela['minutos'].iloc[i], tabela['ap_persistencia'].iloc[i]),
                    xytext=(0, 6), textcoords='offset points', ha='center', fontsize=8, color='#2a78d6')
    ax.set_title('Persistencia: quanto o score da camera agora preve a pluma k minutos depois')
    ax.set_xlabel('Minutos no futuro (cabeca)')
    ax.set_ylabel('AP')
    ax.set_ylim(0, None)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(caminho, dpi=150, bbox_inches='tight')
    plt.show()


def plot_forecasts(ds, previsoes, previsores, cabecas, inicio, dias, caminho):
    #For each head: the forecast made at t is drawn at the instant it refers to (t + k),
    #next to the real score of the camera at that instant (gray). Good forecast = blue rises with the gray
    fim = pd.Timestamp(inicio) + pd.Timedelta(days=dias)
    real = ds[(ds['Data'] >= pd.Timestamp(inicio)) & (ds['Data'] < fim)].sort_values('Data')
    fig, eixos = plt.subplots(len(cabecas), 1, figsize=(11, 2.4*len(cabecas)), sharex=True, squeeze=False)
    cont = 0
    for minutos in cabecas:
        ax = eixos[cont, 0]
        ax.plot(real['Data'], real['y'], color='#999999', linewidth=1.2, label='score real da camera')
        instante = previsoes['Data'] + pd.Timedelta(minutes=minutos)
        mascara = (instante >= pd.Timestamp(inicio)) & (instante < fim)
        ax.plot(instante[mascara], previsoes['prob_' + str(minutos)][mascara], color='#2a78d6', linewidth=1.6,
                label='previsao feita ' + str(minutos) + ' min antes')
        ax.axhline(0.5, color='#555555', linewidth=0.8, linestyle=':')
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel(str(minutos) + ' min\n(' + previsores[minutos] + ')', fontsize=9)
        ax.legend(frameon=False, fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        cont = cont+1
    eixos[0, 0].set_title('Previsao de cada cabeca no instante a que ela se refere (dias do hold out, nunca vistos no treino)')
    fig.tight_layout()
    fig.savefig(caminho, dpi=150, bbox_inches='tight')
    plt.show()


def plot_precision_recall(resultados, tabela, previsores, caminho):
    #Precision x recall curve of each head in the TEST: blue = final system, gray dashed = persistence,
    #dotted = prevalence (a random guess). The point = the threshold chosen in the train, applied to the test
    from sklearn.metrics import precision_recall_curve
    cabecas = list(resultados.keys())
    colunas = 4
    n_linhas = (len(cabecas) + colunas - 1) // colunas
    fig, eixos = plt.subplots(n_linhas, colunas, figsize=(14, 3.6*n_linhas), squeeze=False)
    cont = 0
    for minutos in cabecas:
        ax = eixos[cont // colunas, cont % colunas]
        dx = resultados[minutos]
        linha = tabela[tabela['minutos'] == minutos].iloc[0]
        precisao, recall, limiares = precision_recall_curve(dx['y_true'], dx['sistema'])
        ax.plot(recall, precisao, color='#2a78d6', linewidth=1.8, label='sistema final')
        if previsores[minutos] != 'persistencia':
            precisao, recall, limiares = precision_recall_curve(dx['y_true'], dx['persistencia'])
            ax.plot(recall, precisao, color='#999999', linewidth=1.2, linestyle='--', label='persistencia')
        ax.axhline(linha['prevalencia'], color='#555555', linewidth=0.8, linestyle=':')
        ax.plot(linha['recall'], linha['precisao'], 'o', color='#eb6834', markersize=8, label='limiar escolhido')
        ax.annotate(f"P={linha['precisao']:.2f} R={linha['recall']:.2f}", (linha['recall'], linha['precisao']),
                    xytext=(6, 6), textcoords='offset points', fontsize=8, color='#333333')
        ax.set_title(str(minutos) + ' min (' + previsores[minutos] + ') | AP ' + f"{linha['ap']:.2f}", fontsize=10)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel('Recall', fontsize=9)
        ax.set_ylabel('Precisao', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if cont == 0:
            ax.legend(frameon=False, fontsize=8, loc='lower left')
        cont = cont+1
    fig.suptitle('Precisao x recall no conjunto de teste (hold out)', fontsize=12)
    fig.tight_layout()
    fig.savefig(caminho, dpi=150, bbox_inches='tight')
    plt.show()
