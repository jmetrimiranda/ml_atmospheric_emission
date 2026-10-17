# Databricks notebook source
# MAGIC %md
# MAGIC # Pluma: previsao e recomendacao a cada 5 minutos
# MAGIC
# MAGIC Roda como **Job a cada 5 minutos**. Le as linhas novas do PIMS (`bronze.sch_pims.tb_dados_pims_parquet`), faz a
# MAGIC pivotagem (uma coluna por tag, como no Excel do treino), usa os modelos ja treinados (`models/`) e grava duas tabelas Delta:
# MAGIC
# MAGIC 1. **previsao**: uma linha por instante de 5 min: `Data`, `score_camera`, `prob_<k>` e `alarme_<k>` de cada cabeca, `versao_modelo`;
# MAGIC 2. **recomendacao**: uma linha por tag a mudar, nos instantes com alarme: `Data`, `risco_atual`, `risco_recomendado`,
# MAGIC    `variavel`, `valor_atual`, `valor_recomendado`, `direcao`, `validada` (False: recomendacao NAO validada), `versao_modelo`.
# MAGIC
# MAGIC O PIMS vem no formato longo (uma linha por tag: `Tag`, `DtHora`, `Valor`, `dtInsert`, `dt_arq`) e tem as tags com os
# MAGIC mesmos nomes do Excel (as 6 tags do modelo e `nn_score_smog`). O `DtHora` ja e o horario da planta (o `+00:00` e so
# MAGIC como o Databricks mostra), por isso a sessao fica em UTC: assim o horario nao muda e bate com o Excel.
# MAGIC
# MAGIC Rodar de novo nao duplica linhas (MERGE pela chave).
# MAGIC
# MAGIC **Instalacao (proxima celula):** com Git folder (repositorio clonado no Databricks), troque `<voce>` pelo seu usuario
# MAGIC e use essa mesma pasta no widget `pasta`. Com Volume: o caminho do `.whl` e a pasta do Volume (com `models/` e `data/`).

# COMMAND ----------

# MAGIC %pip install /Workspace/Users/<voce>/ml_atmospheric_emission

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text('pasta', '/Workspace/Users/<voce>/ml_atmospheric_emission', '1. Pasta do repositorio (models/ e data/)')
dbutils.widgets.text('tabela_pims', 'bronze.sch_pims.tb_dados_pims_parquet', '2. Tabela do PIMS (formato longo)')
dbutils.widgets.text('tabela_previsao', 'produtos_dev.sch_predicao_plumas.pluma_previsao_5min', '3. Tabela de previsao (saida)')
dbutils.widgets.text('tabela_recomendacao', 'produtos_dev.sch_predicao_plumas.pluma_recomendacao_5min', '4. Tabela de recomendacao (saida)')
dbutils.widgets.text('dias_primeira_vez', '7', '5. Dias lidos na primeira execucao')
dbutils.widgets.dropdown('treinar', 'nao', ['nao', 'sim'], '6. Treinar de novo os modelos no cluster')

# COMMAND ----------

import os
from datetime import timedelta

import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.window import Window

pasta = dbutils.widgets.get('pasta')
nome_pims = dbutils.widgets.get('tabela_pims')
nome_previsao = dbutils.widgets.get('tabela_previsao')
nome_recomendacao = dbutils.widgets.get('tabela_recomendacao')
dias_primeira_vez = int(dbutils.widgets.get('dias_primeira_vez'))

#DtHora of the PIMS is already the time of the plant (like the Excel), only shown as +00:00
#With the session in UTC the time does not change when going to pandas and back
spark.conf.set('spark.sql.session.timeZone', 'UTC')

#PLUMA_PASTA BEFORE importing the package: it reads models/ and data/ from there
os.environ['PLUMA_PASTA'] = pasta
from ml_atmospheric_emission.config import TAGS
from ml_atmospheric_emission.pipeline import tabela_previsao, tabela_recomendacao
from ml_atmospheric_emission.modeling import train

# COMMAND ----------

#Optional: makes the .pkl again IN the cluster (with the versions of numpy, scikit-learn and LightGBM of the cluster)
#Uses data/processed/dataset_processed.csv of the pasta and takes a few seconds
if dbutils.widgets.get('treinar') == 'sim':
    train.main()

# COMMAND ----------

#Last instant already saved (None in the first run). As text, so the time zone does not change it
ultimo = None
if spark.catalog.tableExists(nome_previsao):
    texto = spark.table(nome_previsao).agg(F.date_format(F.max('Data'), 'yyyy-MM-dd HH:mm:ss')).collect()[0][0]
    if texto is not None:
        ultimo = pd.Timestamp(texto)

#Reads from 60 minutes before the last instant (each instant needs the 30 minutes before it)
#In the first run: the last days of the PIMS (widget dias_primeira_vez)
if ultimo is None:
    texto = spark.table(nome_pims).filter(F.col('Tag') == 'nn_score_smog').agg(F.date_format(F.max('DtHora'), 'yyyy-MM-dd HH:mm:ss')).collect()[0][0]
    inicio = pd.Timestamp(texto) - timedelta(days=dias_primeira_vez)
else:
    inicio = ultimo - timedelta(minutes=60)

#Only the tags of the model and the camera score (the same names of the Excel)
colunas_pims = TAGS + ['nn_score_smog']
longo = spark.table(nome_pims)
longo = longo.filter(F.col('Tag').isin(colunas_pims))
longo = longo.filter(F.col('DtHora') > F.lit(str(inicio)).cast('timestamp'))

#Valor comes as text: becomes a number (a decimal comma becomes a point)
longo = longo.withColumn('Valor', F.regexp_replace('Valor', ',', '.').cast('double'))

#If the same tag and instant came more than once, keeps the last inserted
janela = Window.partitionBy('Tag', 'DtHora').orderBy(F.desc('dtInsert'))
longo = longo.withColumn('ordem', F.row_number().over(janela)).filter(F.col('ordem') == 1)

#Pivot: one row per instant, one column per tag (like the Excel of the training)
largo = longo.groupBy('DtHora').pivot('Tag', colunas_pims).agg(F.first('Valor'))

#To pandas with the time as text (the time zone does not change it), then back to date in pandas
largo = largo.withColumn('Data', F.date_format('DtHora', 'yyyy-MM-dd HH:mm:ss')).drop('DtHora')
ds_bruto = largo.toPandas()
ds_bruto['Data'] = pd.to_datetime(ds_bruto['Data'])
ds_bruto = ds_bruto[['Data'] + colunas_pims].sort_values('Data').reset_index(drop=True)
print('Lido do PIMS desde', inicio, '| instantes:', len(ds_bruto), '| ultimo instante ja salvo:', ultimo)
if len(ds_bruto) > 0:
    print('Ultimo instante do PIMS:', ds_bruto['Data'].max())

# COMMAND ----------

def salvar(ds, nome, chaves):
    #Writes the new rows in the Delta table without duplicating (MERGE by the keys): running again is safe
    if len(ds) == 0:
        print(nome, ': nada novo')
        return
    #Data goes as text and becomes timestamp in Spark (session in UTC): the time of the plant does not change
    ds = ds.copy()
    ds['Data'] = ds['Data'].dt.strftime('%Y-%m-%d %H:%M:%S')
    novos = spark.createDataFrame(ds).withColumn('Data', F.col('Data').cast('timestamp'))
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
