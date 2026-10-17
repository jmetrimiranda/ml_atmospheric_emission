# Databricks notebook source
# MAGIC %md
# MAGIC # Pluma: previsao e recomendacao a cada 5 minutos
# MAGIC
# MAGIC Roda como **Job a cada 5 minutos**. Le as linhas novas da tabela crua (mesmas colunas de `data/raw`), usa os modelos
# MAGIC ja treinados (`models/` no Volume) e grava duas tabelas Delta:
# MAGIC
# MAGIC 1. **previsao**: uma linha por instante de 5 min: `Data`, `score_camera`, `prob_<k>` e `alarme_<k>` de cada cabeca, `versao_modelo`;
# MAGIC 2. **recomendacao**: uma linha por tag a mudar, nos instantes com alarme: `Data`, `risco_atual`, `risco_recomendado`,
# MAGIC    `variavel`, `valor_atual`, `valor_recomendado`, `direcao`, `validada` (False: recomendacao NAO validada), `versao_modelo`.
# MAGIC
# MAGIC Pasta no Volume (widget `pasta`):
# MAGIC ```
# MAGIC <pasta>/ml_atmospheric_emission-1.1.0-py3-none-any.whl
# MAGIC <pasta>/models/modelo_y_90.pkl ... modelo_y_360.pkl, limiares.json
# MAGIC <pasta>/data/processed/dataset_processed.csv   (historico: limites da recomendacao)
# MAGIC ```
# MAGIC Rodar de novo nao duplica linhas (MERGE pela chave).
# MAGIC
# MAGIC **Instalacao (proxima celula):** com Volume, o caminho do `.whl`; com Git folder (repositorio clonado no Databricks),
# MAGIC troque por `%pip install /Workspace/Users/<voce>/ml_atmospheric_emission` e use essa pasta no widget `pasta`.

# COMMAND ----------

# MAGIC %pip install /Volumes/catalogo/schema/pluma/ml_atmospheric_emission-1.1.0-py3-none-any.whl

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text('pasta', '/Volumes/catalogo/schema/pluma', '1. Pasta no Volume (models/ e data/)')
dbutils.widgets.text('tabela_bruta', 'catalogo.schema.tags_pluma_bruto', '2. Tabela crua (mesmas colunas de data/raw)')
dbutils.widgets.text('coluna_tempo', 'Data', '3. Coluna de tempo da tabela crua')
dbutils.widgets.text('tabela_previsao', 'catalogo.schema.pluma_previsao', '4. Tabela de previsao (saida)')
dbutils.widgets.text('tabela_recomendacao', 'catalogo.schema.pluma_recomendacao', '5. Tabela de recomendacao (saida)')
dbutils.widgets.dropdown('treinar', 'nao', ['nao', 'sim'], '6. Treinar de novo os modelos no cluster')

# COMMAND ----------

import os
from datetime import timedelta

from pyspark.sql import functions as F

pasta = dbutils.widgets.get('pasta')
nome_bruta = dbutils.widgets.get('tabela_bruta')
coluna_tempo = dbutils.widgets.get('coluna_tempo')
nome_previsao = dbutils.widgets.get('tabela_previsao')
nome_recomendacao = dbutils.widgets.get('tabela_recomendacao')

#The time of the plant is local (UTC-3), like in the Excel
spark.conf.set('spark.sql.session.timeZone', 'America/Sao_Paulo')

#PLUMA_PASTA BEFORE importing the package: it reads models/ and data/ from there
os.environ['PLUMA_PASTA'] = pasta
from ml_atmospheric_emission.pipeline import tabela_previsao, tabela_recomendacao
from ml_atmospheric_emission.modeling import train

# COMMAND ----------

#Optional: makes the .pkl again IN the cluster (with the versions of numpy, scikit-learn and LightGBM of the cluster)
#Uses data/processed/dataset_processed.csv of the Volume and takes a few seconds
if dbutils.widgets.get('treinar') == 'sim':
    train.main()

# COMMAND ----------

#Last instant already saved (None in the first run)
ultimo = None
if spark.catalog.tableExists(nome_previsao):
    ultimo = spark.table(nome_previsao).agg(F.max('Data')).collect()[0][0]

#Raw rows: from 60 minutes before the last instant (each instant needs the 30 minutes before it)
bruto = spark.table(nome_bruta)
if ultimo is not None:
    bruto = bruto.filter(F.col(coluna_tempo) > F.lit(ultimo - timedelta(minutes=60)))
ds_bruto = bruto.toPandas().rename(columns={coluna_tempo: 'Data'})
print('Linhas cruas lidas:', len(ds_bruto), '| ultimo instante ja salvo:', ultimo)

# COMMAND ----------

def salvar(ds, nome, chaves):
    #Writes the new rows in the Delta table without duplicating (MERGE by the keys): running again is safe
    if len(ds) == 0:
        print(nome, ': nada novo')
        return
    novos = spark.createDataFrame(ds)
    if not spark.catalog.tableExists(nome):
        novos.write.saveAsTable(nome)
    else:
        novos.createOrReplaceTempView('novos')
        condicao = []
        for chave in chaves:
            condicao.append('t.`' + chave + '` = s.`' + chave + '`')
        spark.sql('MERGE INTO ' + nome + ' t USING novos s ON ' + ' AND '.join(condicao) + ' WHEN NOT MATCHED THEN INSERT *')
    print(nome, ': linhas gravadas', len(ds))

# COMMAND ----------

#1. Forecast of each new instant
previsao = tabela_previsao(ds_bruto, a_partir_de=ultimo)
salvar(previsao, nome_previsao, ['Data'])

# COMMAND ----------

#2. Recommendation of each new instant with alarm (in the first run: only the last day, to be quick)
inicio_recomendacao = ultimo
if inicio_recomendacao is None and len(ds_bruto) > 0:
    inicio_recomendacao = ds_bruto['Data'].max() - timedelta(days=1)
recomendacao = tabela_recomendacao(ds_bruto, a_partir_de=inicio_recomendacao)
salvar(recomendacao, nome_recomendacao, ['Data', 'variavel'])

# COMMAND ----------

display(spark.table(nome_previsao).orderBy(F.desc('Data')).limit(12))

# COMMAND ----------

if spark.catalog.tableExists(nome_recomendacao):
    display(spark.table(nome_recomendacao).orderBy(F.desc('Data')).limit(24))
