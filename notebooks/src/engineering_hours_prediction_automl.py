# Databricks notebook source
from pyspark.sql.functions import col, element_at
from sklearn.preprocessing import PowerTransformer
import numpy as np
import joblib
import mlflow
from pyspark.sql.functions import col, element_at
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col
import matplotlib.pyplot as plt


# COMMAND ----------

filepath=["dbfs:/FileStore/Imacs/df_data_21.csv"]
raw_df = spark.read.csv(filepath, sep=",", header=True, inferSchema=True)
raw_df = (
    raw_df
    .drop('ProjectNo', 'industry_codes')
    .filter(col('total_sell_price') <= 1E7)
    .filter(col('hours') > 0)
    .withColumnRenamed("hours", "Hours")
)

# COMMAND ----------

display(raw_df.count())
display(raw_df.printSchema())

# COMMAND ----------

# Convert to Pandas and Apply Box-Cox
pdf = raw_df.toPandas()


# COMMAND ----------

from scipy.special import boxcox, inv_boxcox

# Ensure target is positive
assert (pdf['Hours'] > 0).all(), "Box-Cox requires all positive values!"

pt = PowerTransformer(method='box-cox')
pt.fit(pdf[['Hours']])

print(pt.lambdas_)

pdf['hours_transformed'] = pt.transform(pdf[['Hours']])

# Save transformer to use later
joblib.dump(pt, "/dbfs/tmp/boxcox_hours.pkl")

# Step 3: Convert back to Spark DataFrame and Split
transformed_df = spark.createDataFrame(pdf.drop(columns=["Hours"]))

# COMMAND ----------

pdf_2 = pdf.copy()
pdf_2['Hours'] = pdf_2['hours_transformed']
pdf_2 = pdf_2.drop(columns=['hours_transformed'])

pdf_2['hours_check'] = pt.inverse_transform(pdf_2[['Hours']])


display(pdf)

# COMMAND ----------

train_df, test_df = transformed_df.randomSplit([.8, .2], seed=42)
test_df.count()

# COMMAND ----------

from databricks import automl
summary = automl.regress(train_df, target_col="hours_transformed", primary_metric="rmse", max_trials=5 ,timeout_minutes=20)

# COMMAND ----------

print(summary.best_trial)

# COMMAND ----------

# Load the best trial as an MLflow Model

model_uri = f"runs:/{summary.best_trial.mlflow_run_id}/model"
predict = mlflow.pyfunc.spark_udf(spark, model_uri)

pred_df = test_df.withColumn("prediction", predict(*test_df.drop("hours").columns))
pred_df = pred_df.withColumn("prediction", element_at(col("prediction"), 1))
display(pred_df)


# COMMAND ----------

# Convert predictions to Pandas for inverse transform
pred_pdf = pred_df.select("prediction", "hours_transformed").toPandas()

# Load transformer
pt = joblib.load('/dbfs/tmp/boxcox_hours.pkl')

# 🔧 Rename columns to match original 'Hours' feature name used during fit
prediction_input = pred_pdf[["prediction"]].copy()
prediction_input.columns = ["Hours"]

hours_transformed_input = pred_pdf[["hours_transformed"]].copy()
hours_transformed_input.columns = ["Hours"]

# Apply inverse Box-Cox transformation
pred_pdf["prediction_original"] = pt.inverse_transform(prediction_input)
pred_pdf["hours_original"] = pt.inverse_transform(hours_transformed_input)

# Display final comparison
display(pred_pdf[["hours_original", "prediction_original", "hours_transformed", "prediction"]])


# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col

# Evaluate in Box-Cox transformed space (Spark RMSE)
regression_evaluator = RegressionEvaluator(
    predictionCol="prediction", 
    labelCol="hours_transformed", 
    metricName="rmse"
)
pred_df_fixed = pred_df.withColumn("prediction", col("prediction"))
rmse = regression_evaluator.evaluate(pred_df_fixed)
print(f"RMSE on test dataset (Box-Cox transformed): {rmse:.3f}")

# Evaluate in original space (inverse transformed)
from sklearn.metrics import root_mean_squared_error

rmse_original = root_mean_squared_error(pred_pdf["hours_original"], pred_pdf["prediction_original"])
print(f"RMSE on test dataset (after inverse Box-Cox): {rmse_original:.3f}")


# COMMAND ----------

plt.figure(figsize=(8, 6))
plt.scatter(pred_pdf["hours_original"], pred_pdf["prediction_original"], alpha=0.6)
plt.plot([pred_pdf["hours_original"].min(), pred_pdf["hours_original"].max()],
         [pred_pdf["hours_original"].min(), pred_pdf["hours_original"].max()],
         color='red', linestyle='--')
plt.xlabel("Actual Hours")
plt.ylabel("Predicted Hours")
plt.title("Actual vs Predicted (Original Scale)")
plt.grid(True)
plt.show()

# COMMAND ----------

import matplotlib.pyplot as plt
# Get residuals from the regression pred_pdf
pred_pdf['residuals'] = pred_pdf['hours_original'] - pred_pdf['prediction_original']

# Plot residuals vs fitted values (prediction)
plt.scatter(pred_pdf['prediction_original'], pred_pdf['residuals'])
plt.xlabel('Predicted Hours')
plt.ylabel('Residuals')
plt.title('Residuals vs. Predicted Values')
plt.axhline(y=0, color='black', linestyle='--')
plt.show()
