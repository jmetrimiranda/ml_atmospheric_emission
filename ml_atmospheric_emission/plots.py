import matplotlib.pyplot as plt


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
