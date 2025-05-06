# Databricks notebook source
filepath=["dbfs:/FileStore/Imacs/df_data_17.csv"]
raw_df = spark.read.csv(filepath, sep=",", header=True, inferSchema=True)
raw_df = raw_df.drop('ProjectNo')

# COMMAND ----------

display(raw_df.count())
display(raw_df.printSchema())

# COMMAND ----------

train_df, test_df = raw_df.randomSplit([.8, .2], seed=42)

# COMMAND ----------

import mlflow
mlflow.end_run

# COMMAND ----------

from databricks import automl
summary = automl.regress(train_df, target_col="Hours", primary_metric="rmse", max_trials=5 ,timeout_minutes=20)

# COMMAND ----------

print(summary.best_trial)

# COMMAND ----------

# Load the best trial as an MLflow Model
import mlflow

model_uri = f"runs:/{summary.best_trial.mlflow_run_id}/model"

predict = mlflow.pyfunc.spark_udf(spark, model_uri)

pred_df = test_df.withColumn("prediction", predict(*test_df.drop("hours").columns))

from pyspark.sql.functions import col, element_at

pred_df = pred_df.withColumn("prediction", element_at(col("prediction"), 1))

display(pred_df)

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

regression_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="hours", metricName="rmse")

from pyspark.sql.functions import col

pred_df_fixed = pred_df.withColumn("prediction", col("prediction"))

rmse = regression_evaluator.evaluate(pred_df_fixed)
print(f"RMSE on test dataset: {rmse:.3f}")

# COMMAND ----------


