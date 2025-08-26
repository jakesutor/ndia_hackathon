# Databricks notebook source
import requests
import pandas as pd
from io import BytesIO

url = "https://github.com/AED-KM/ndia_hackathon/raw/main/data/usaf_pilots_synthetic.xlsx"
response = requests.get(url)

aeromedical_class_history = pd.read_excel(BytesIO(response.content), sheet_name="aeromedical_class_history")
encounters = pd.read_excel(BytesIO(response.content), sheet_name="encounters")
hospitalizations = pd.read_excel(BytesIO(response.content), sheet_name="hospitalizations")
immunizations = pd.read_excel(BytesIO(response.content), sheet_name="immunizations")
labs = pd.read_excel(BytesIO(response.content), sheet_name="labs")
mental_health = pd.read_excel(BytesIO(response.content), sheet_name="mental_health")
pilots = pd.read_excel(BytesIO(response.content), sheet_name="pilots")
predictive_labels_monthly = pd.read_excel(BytesIO(response.content), sheet_name="predictive_labels_monthly")
profiles = pd.read_excel(BytesIO(response.content), sheet_name="profiles")
readiness_kpis_monthly = pd.read_excel(BytesIO(response.content), sheet_name="readiness_kpis_monthly")

# COMMAND ----------

# Convert Pandas DataFrames to Spark DataFrames
aeromedical_class_history_df = spark.createDataFrame(aeromedical_class_history)
encounters_df = spark.createDataFrame(encounters)
hospitalizations_df = spark.createDataFrame(hospitalizations)
immunizations_df = spark.createDataFrame(immunizations)
labs_df = spark.createDataFrame(labs)
mental_health_df = spark.createDataFrame(mental_health)
pilots_df = spark.createDataFrame(pilots)
predictive_labels_monthly_df = spark.createDataFrame(predictive_labels_monthly)
profiles_df = spark.createDataFrame(profiles)
readiness_kpis_monthly_df = spark.createDataFrame(readiness_kpis_monthly)


# COMMAND ----------

hospitalizations_df = hospitalizations_df.withColumnRenamed("admit_date", "date").withColumnRenamed("reason","hospitalization_reason")
profiles_df = profiles_df.withColumnRenamed("start_date","date")
mental_health_df = mental_health_df.withColumnRenamed("medication","mental_health_medication")

# COMMAND ----------

from pyspark.sql.functions import col, date_format

# Function to add year_month column if date column exists and year_month does not exist
# Also rename the date column with the prefix of the table name if date exists in the dataframe
def add_year_month_column(df, table_name):
    if "date" in df.columns:
        df = df.withColumnRenamed("date", f"{table_name}_date")
        if "year_month" not in df.columns:
            df = df.withColumn("year_month", date_format(col(f"{table_name}_date"), "yyyy-MM"))
    return df

# Add year_month column to all tables except pilots_df
aeromedical_class_history_df = add_year_month_column(aeromedical_class_history_df, "aeromedical_class_history")
encounters_df = add_year_month_column(encounters_df, "encounters")
hospitalizations_df = add_year_month_column(hospitalizations_df, "hospitalizations")
immunizations_df = add_year_month_column(immunizations_df, "immunizations")
labs_df = add_year_month_column(labs_df, "labs")
mental_health_df = add_year_month_column(mental_health_df, "mental_health")
predictive_labels_monthly_df = add_year_month_column(predictive_labels_monthly_df, "predictive_labels_monthly")
profiles_df = add_year_month_column(profiles_df, "profiles")
readiness_kpis_monthly_df = add_year_month_column(readiness_kpis_monthly_df, "readiness_kpis_monthly")

# COMMAND ----------

from pyspark.sql.functions import to_date

# Drop the "age" field from the pilots_df table
pilots_df = pilots_df.drop("age","flight_hours_last_12mo")

# Convert pha_last_date from timestamp to date
pilots_df = pilots_df.withColumn("pha_last_date", to_date("pha_last_date"))

# Joining predictive_labels_monthly_df to pilots_df on the column "pilot_id"
merged_df = pilots_df.join(predictive_labels_monthly_df, on="pilot_id", how="left")

# Join readiness_kpis_monthly_df to the merged dataset on "base" and "year_month"
merged_df = merged_df.join(readiness_kpis_monthly_df, on=["base", "year_month"], how="left")

#display(merged_df)

# COMMAND ----------

#merged_df.display()

# COMMAND ----------

# Update each dataframe to be joined with the merged_df on "pilot_id" and "year_month"
aeromedical_class_history_df = merged_df.join(aeromedical_class_history_df, on=["pilot_id", "year_month"], how="left").withColumnRenamed("aeromedical_class_history_date", "date")
encounters_df = merged_df.join(encounters_df, on=["pilot_id", "year_month"], how="left").withColumnRenamed("encounters_date", "date")
hospitalizations_df = merged_df.join(hospitalizations_df, on=["pilot_id", "year_month"], how="left").withColumnRenamed("hospitalizations_date", "date")
immunizations_df = merged_df.join(immunizations_df, on=["pilot_id", "year_month"], how="left").withColumnRenamed("immunizations_date", "date")
labs_df = merged_df.join(labs_df, on=["pilot_id", "year_month"], how="left").withColumnRenamed("labs_date", "date")
mental_health_df = merged_df.join(mental_health_df, on=["pilot_id", "year_month"], how="left").withColumnRenamed("mental_health_date", "date")
profiles_df = merged_df.join(profiles_df, on=["pilot_id", "year_month"], how="left").withColumnRenamed("profiles_date", "date")

# COMMAND ----------

# Perform a full outer join on all matching columns between aeromedical_class_history_df and encounters_df
matching_columns = [col for col in aeromedical_class_history_df.columns if col in encounters_df.columns]
joined_df = aeromedical_class_history_df.join(encounters_df, on=matching_columns, how="full_outer")

#display(joined_df)

# COMMAND ----------

full_joined_df = joined_df.join(hospitalizations_df, on=matching_columns, how="full_outer") \
                     .join(immunizations_df, on=matching_columns, how="full_outer") \
                     .join(labs_df, on=matching_columns, how="full_outer") \
                     .join(mental_health_df, on=matching_columns, how="full_outer") \
                     .join(profiles_df, on=matching_columns, how="full_outer")

# COMMAND ----------

full_joined_df.display()



# COMMAND ----------

# Write merged_df to a Delta table
full_joined_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("fmddt_catalog.gao.all_pilots_data")