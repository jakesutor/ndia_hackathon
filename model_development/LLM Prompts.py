# Databricks notebook source
# MAGIC %pip install openai

# COMMAND ----------

# MAGIC %pip install --upgrade typing_extensions

# COMMAND ----------

# Set up connection
from typing import List
from openai import OpenAI
from pyspark.sql.functions import col

DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

client = OpenAI(
    api_key=DATABRICKS_TOKEN,
    base_url="https://dbc-7e282311-5ce7.cloud.databricks.com/serving-endpoints"
)

# COMMAND ----------

# Create function for response
def generate_response(question):
    response = client.chat.completions.create(
        model="databricks-claude-3-7-sonnet",
        messages=[
            {
                "role": "user",
                "content": question
            }
        ],
        max_tokens=5000
    )
    return response.choices[0].message.content

# COMMAND ----------

# Load dataframe
df = spark.table("fmddt_catalog.gao.all_pilots_data")

# COMMAND ----------

# Filter the dataframe to a single pilot_id
pilot_id = "P0202"  # Replace with the actual pilot_id
filtered_df = df.filter(df.pilot_id == pilot_id)

display(filtered_df)

# COMMAND ----------

# Test analysis of one pilot
pilot_data = filtered_df.collect()[0].asDict()
question = f"Provide a brief description of Pilot {pilot_data['pilot_id']}. Describe his health trends, age, flight hours, and medical history. Provide a brief analysis on his likelihood of having a medical event in the next 90 days. Describe how likely it is that he would have a disqualifying medical event. Provide justification for all analyses including identifying the fields you draw your analysis from. Data: {pilot_data}"
print(generate_response(question))

# COMMAND ----------

desired_columns = ['pilot_id', 
                   'year_month', 'age', 'flight_hours_last_12mo', 'abnormal_labs_6mo', 'encounters_6mo', 'profile_next_90d',
                   'classification', 'visit_type','icd10_desc','cpt_desc', 'hospitalization_reason','vaccine', 'status', 'panel', 'test_name',
                   'abnormal_flag', 'diagnosis','therapy_sessions', 'reason']

# COMMAND ----------

# Test full dataset for positive profile_next_90d
full_data = df.selectExpr(*desired_columns).filter(col('profile_next_90d')==1).collect()
new_question = f"Analyzing the dataset {full_data}, provide a brief overview of what fields appear correlated with having a profile_next_90d. A 1 in this field suggests the individual is likely to have a potential disqualifying event in the next 90 days. What medical events or history likely have the most impact on this? Provide specific pilot_ids as examples. What fields should be included in a ML model to predict the probability of this ocurring?"
print(generate_response(new_question))

# COMMAND ----------

# Test against one pilot
pilot_data_test = df.selectExpr(*desired_columns).filter(col('pilot_id')=='P0001').collect()
new_question = f"Using the prior information about which elements area most likely correlated with issues, analyze pilot P0001 from the dataset {pilot_data_test} and provide a score of how likely this pilot is to have a disqualifying event in the next 90 days. Provide justification"
print(generate_response(new_question))
