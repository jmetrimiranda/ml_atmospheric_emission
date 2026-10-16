import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def convert_to_ordinal(row):
    if row['nn_score_smog'] >= 0.5:
        return 1
    else:
        return 0

def linear_graph(x,y,titulo):

    fig, ax = plt.subplots()

    # 3. Plotar a linha com customizações (cor, estilo, marcadores)
    ax.plot(x, y, color='blue', linestyle='--', marker='o', linewidth=2, label='Desempenho')

    # 4. Adicionar títulos e rótulos aos eixos
    ax.set_title(titulo)
    ax.set_xlabel(f'Eixo X : {str(x.name)}')
    ax.set_ylabel(f'Eixo Y: {str(y.name)}')

    # 5. Adicionar grade e legenda
    ax.grid(True)
    ax.legend()

    # 6. Exibir o gráfico na tela
    plt.show()
