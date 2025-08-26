# Databricks notebook source
from helper_functions import *
import pandas as pd
import numpy as np
import random 
from pyspark.sql.functions import kurtosis
import sys
import json
import os
from datetime import date

from pyspark.sql.functions import *
from pyspark.sql import Window, DataFrame
from pyspark.sql.functions import regexp_replace
from pyspark.sql.types import StructType

from pyspark.ml.classification import GBTClassifier, LogisticRegression,RandomForestClassifier

from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, VectorIndexer
from pyspark.ml import Pipeline, PipelineModel

from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

try:
  import mlflow
  import mlflow.spark as mlflow_sklearn
except:
  !pip install mlflow
  import mlflow
  import mlflow.spark as mlflow_sklearn

spark.conf.set("spark.databricks.io.cache.enabled","true")
spark.conf.set("spark.databricks.delta.checkLatestSchemaOnRead", "false")
spark.conf.set('spark.sql.shuffle.partitions', 'auto')  
import warnings
warnings.filterwarnings('ignore')


# COMMAND ----------

# Step 2.1 - Set Parameters For Model

dbutils.widgets.text('Training Date', '')

dbutils.widgets.dropdown('Model Performance Metric', 'LogLoss', ['LogLoss', 'Weighted LogLoss', 'Balanced_Accuracy', 
                                                                 'Accuracy', 'F1_Score', 'Recall', 'Precision'])

dbutils.widgets.dropdown('Maximum Model Training Sample Size', '1', ['0.5', '0.6', '0.7', '0.8', '0.9','1'])

dbutils.widgets.dropdown('Use MLFlow', 'No', ['Yes', 'No'])

dbutils.widgets.text('User Profile Number', '')

dbutils.widgets.dropdown('Run As Main', 'Yes', ['Yes', 'No'])

dbutils.widgets.dropdown('Model Type', 'LR', ['LR', 'xgboost','gbtree','rforest'])

# Step 2.2 - Get Parameter Values

training_date = dbutils.widgets.get('Training Date')

if training_date == '':

  training_date = date.today()
  training_date = str(training_date.year).zfill(2) + '-' + str(training_date.month).zfill(2) + '-' + str(training_date.day).zfill(2)

model_performance_metric = dbutils.widgets.get('Model Performance Metric')

maximum_training_set_size = float(dbutils.widgets.get('Maximum Model Training Sample Size'))

use_mlflow = dbutils.widgets.get('Use MLFlow')

use_mlflow = True if use_mlflow == 'Yes' else False

user_profile = dbutils.widgets.get('User Profile Number')

run_as_main = dbutils.widgets.get('Run As Main')
model_type = dbutils.widgets.get('Model Type')

# COMMAND ----------

class Train_Model():
  
  """

  Trains ML Model For DARQ Risk Modeling Approach

  """
  
  def __init__(self,
               maximum_training_size: float,
               performance_metric: str, 
               training_date: str, 
               use_mlflow: bool,
               model_type:str,
               initial_training_size: float = 0.80,
               id: str = 'pilot_id',
               target: str = "profile_next_90d",
               columns_to_exclude_from_feature_set: list = ['pilot_id', 'year_month'],
               model_name: str = 'flight_risk_classification_model',
               champion_model_performance_metrics_path: str = r"fmddt_catalog.gao.predictive_labels_monthly_model_champion_metrics_v1",
               model_performance_metrics_path: str = r"fmddt_catalog.gao.predictive_labels_monthly_model_metrics_V1",
               repartition_number: int = 512):
    self.cat_features = ['gender','aircraft','aeromedical_class_current','dental_readiness','pha_status']

    self.id = id 
    
    self.count_vec = []
    
    self.num_features  = ["age","flight_hours_last_12mo","abnormal_labs_6mo","encounters_6mo","pha_overdue_flag","flight_hours_total","immunization_compliance"]
    self.target = target
    self.use_mlflow = use_mlflow
    self.training_date = training_date
    self.training_date_file_name = training_date.replace('-', '_')
    self.model_type = model_type
    self.maximum_train_size = initial_training_size
    self.maximum_test_size = 1 - initial_training_size
    self.all_data_size = maximum_training_size
    self.champion_model_performance_metrics_path = champion_model_performance_metrics_path
    self.model_performance_metrics_path =  model_performance_metrics_path
    self.repartition_number = repartition_number
    self.feature_engineering_path = 'dbfs:/FileStore/models/feature_engineering'

    self.model_name = f"{model_name}_{model_type}"
    
    self.model_path = F"dbfs:/FileStore/models/{model_name}_{self.training_date_file_name}"
    
    self.bso_info = "fmddt_catalog.gao.all_pilots_data"#"fmddt_catalog.gao.predictive_labels_monthly"
   # self.bso_info_index = self.bso_info.index
    
   # self.bso_info.reset_index(drop = True, inplace = True)
    
    try:
      
      self.training_data = spark.table(self.bso_info) 
      
    except:
      
      sys.exit( ' [ WARNING: MODEL DATA DOES NOT EXIST. PLEASE RUN MODULE 2 (FEATURE ENGINEERING) BEFORE RE-RUNNING ] ' )
    
    performance_metric_dictionary = {
                                      'LogLoss': 'logLoss',
                                      'Weighted LogLoss': 'logLoss',
                                      'Balanced_Accuracy': 'weightedRecall',
                                      'Accuracy': 'accuracy',
                                      'F1_Score': 'f1',
                                      'Recall': 'recallByLabel',
                                      'Precision': 'precisionByLabel'
                                     }
    
    self.performance_metric = performance_metric_dictionary[performance_metric]
    
    self.initial_training_size = initial_training_size
    self.test_size = 1 - initial_training_size


    
    if self.use_mlflow:
      
      self._create_experiment_name()
    
    self.performance_metric_results = pd.DataFrame(columns = ['training_date','Model_Type', 'population_type', 'sample_size',
                                                              'recall', 'precision', 'f1', 'npv', 'logloss', 'accuracy', 'balanced_accuracy','tp', 'tn','fp', 'fn'])
    
  def _create_experiment_name(self) -> str:
    
    """
    Creates MLFlow Experiment Name
    
    """
    
    #self.experiment_name = f'{self.bso_to_model}_darq_risk_classification_model_{self.training_date_file_name}_{self.model_type}'

    return self.experiment_name
  def _train_test_split(self): 
    
    ids = self.training_data.select(self.id).distinct()
    
    # get maximum training size 
    if self.all_data_size!=1: 
      ids = ids.sample(self.all_data_size, seed=123)

      
    
    # Split test train 
    self.training_data_ones = self.training_data.filter(col(self.target)== 1)
    self.training_data_zeros = self.training_data.filter(col(self.target)== 0)
    train_ids_ones, test_ids_ones = self.training_data_ones.randomSplit([self.maximum_train_size,self.maximum_test_size],seed=123)
    train_ids_zeros, test_ids_zeros = self.training_data_zeros.randomSplit([self.maximum_train_size,self.maximum_test_size],seed=123)
    self.train = train_ids_ones.union(train_ids_zeros).cache()
    self.test = test_ids_zeros.union(test_ids_ones).cache()
    
  def _create_feature_set(self):
    
    """
    Identifies Feature Set To Train ML Model On, SPLITS BETWEEN TRAIN AND TEST 
    
    """

    self.pipeline_model, self.train, self.feature_names = feature_engineering(data = self.train , cat_features = self.cat_features, 
                                                              num_features = self.num_features,count_vec_features = self.count_vec)
    self.test = self.pipeline_model.transform(self.test)
   
    self.pipeline_model.write() \
                   .overwrite() \
                   .save('dbfs:/FileStore/models/feature_engineering')
    print( f' [ Feature Engineering Pipeline saved as: {'dbfs:/FileStore/models/feature_engineering'} ]')
    
    column_names = [tuple(self.feature_names)]
    column_names = spark.createDataFrame(column_names)

    
    column_names.write.mode('overwrite')\
    .format('parquet')\
    .save('dbfs:/FileStore/models/feature_inventory')
    
    return 
        
  def _train_model(self):
    
    """
    Trains ML Model To Predict Expired Documents Running A Balance
    
    """
    self._train_test_split()
    
    self._create_feature_set()
    
    if self.model_type == 'LR':
      

      self.model = LogisticRegression(featuresCol = 'features', labelCol = self.target)
    
      self.parameter_grid = ParamGridBuilder().addGrid(self.model.regParam, [0.01, 0.5, 1.0, 2.0, 3.0]) \
                                       .addGrid(self.model.elasticNetParam, [0.0, 0.5, 1.0]) \
                                       .addGrid(self.model.maxIter, [1, 5, 10 ]) \
                                       .build()
    
    elif self.model_type == 'gbtree':
      self.model  = GBTClassifier(featuresCol = 'features', labelCol = self.target)

      self.parameter_grid = RandomGridBuilder(36).addDistr(self.model.maxDepth, lambda: np.random.randint(3, 10))\
                             .addDistr(self.model.maxIter,lambda:  np.random.randint(10, 50))\
                             .addDistr(self.model.stepSize,lambda: random.choice(np.arange(0.05,0.1,0.01)))\
                             .addDistr(self.model.validationTol,lambda: 0.01)\
                             .build()   
      
    elif self.model_type=='rforest':
      
      self.model = RandomForestClassifier(featuresCol = 'features', labelCol = self.target)

      self.parameter_grid = RandomGridBuilder(36).addDistr(self.model.maxDepth, lambda: np.random.randint(3, 10))\
                             .addDistr(self.model.numTrees,lambda: np.random.randint(3, 100)) \
                             .build()   
      
      
    elif self.model_type == 'xgboost':
      sc = spark._jsc.sc()

      n_workers = len([executor.host() for executor in sc.statusTracker().getExecutorInfos()])

      print( f'{n_workers} workers available' )
      self.model  = XgboostClassifier(num_workers=n_workers,labelCol= self.target,missing=0.0, use_gpu=True)

      self.parameter_grid = RandomGridBuilder(36).addDistr(self.model.max_depth, lambda: np.random.randint(3, 10))\
                             .addDistr(self.model.n_estimators,lambda: math.floor(150*np.random.power(1))) \
                             .addDistr(self.model.learning_rate,lambda: random.choice(np.arange(0.01,0.1,0.01)))\
                             .addDistr(self.model.reg_lambda, lambda:  random.choice([0.01,0.1,1.0]))\
                             .build()

    self.model_evaluator = MulticlassClassificationEvaluator(metricName = self.performance_metric, predictionCol= 'prediction', labelCol = self.target)
    
    self.crossvalidation_pipeline = CrossValidator(estimator = self.model,
                                                   estimatorParamMaps = self.parameter_grid,
                                                   evaluator = self.model_evaluator,
                                                   numFolds = 2)
    
    
    print('    [ TRAINING CLASSIFICATION MODELS ... ]    ')
    
    if self.use_mlflow:
    
      self._run_mlflow_experiements()
      
    else:
      
      self._run_experiments()
    
    
    
    print('        [ SUCCESSFULLY TRAINED CLASSIFICATION MODEL ...')
    
    print('    [ SELECTING CHAMPION DARQ RISK CLASSIFICATION MODEL ... ]    ')
    
    self._select_champion_model()
    
    print('        [ SELECTED CHAMPION DARQ RISK CLASSIFICATION MODEL ]        ')
    
    return
  
  def _evaluate_model(self, 
                     prediction_data: DataFrame, 
                     population_type: str, 
                     model_run: int, 
                     use_mlflow: bool):
    
    """
    Evaluates ML Models Based On Main Classification Performance Metrics
    
    """
    
    log_loss = self.model_evaluator.evaluate(prediction_data)
    accuracy = self.model_evaluator.evaluate(prediction_data, {self.model_evaluator.metricName: 'accuracy'})
    #f1 = self.model_evaluator.evaluate(prediction_data, {self.model_evaluator.metricName: 'f1'}) I think this calculated f1 for 0's 
    
    _ = prediction_data.select('prediction', self.target)
    
    for column in _.columns:
      
      _ = _.withColumn(column, col(f'{column}').cast('float'))
      
    _ = _.orderBy('prediction')

    _ = _.rdd.map(tuple)
    
    multi_class_metrics = MulticlassMetrics(_)
    cm  = multi_class_metrics.confusionMatrix().toArray()
    tp,tn,fp,fn = cm[1,1], cm[0,0],cm[0,1],cm[1,0]
    precision = multi_class_metrics.precision(label = 1)
    recall = multi_class_metrics.recall(label = 1)
    npv = multi_class_metrics.recall(label = 0)
    balanced_accuracy = (recall + npv) / 2    
    try: 
      f1 = 2*(precision*recall)/(recall+precision)
    except: 
      f1 = 0.0
    
    if use_mlflow == 1:

      mlflow.log_metric(f'{population_type.lower()}_logloss', log_loss) 
      mlflow.log_metric(f'{population_type.lower()}_accuracy', accuracy) 
      mlflow.log_metric(f'{population_type.lower()}_f1', f1) 
      mlflow.log_metric(f'{population_type.lower()}_precision', precision) 
      mlflow.log_metric(f'{population_type.lower()}_recall', recall) 
      mlflow.log_metric(f'{population_type.lower()}_npv', npv) 
      mlflow.log_metric(f'{population_type.lower()}_balanced_accuracy', balanced_accuracy) 

    if population_type == 'Training':
      
      sample_size = self.initial_training_size
      
    elif population_type == 'Test':
      
      sample_size = self.test_size
      
    elif population_type == 'Final':
      
      sample_size = self.all_data_size
      
    new_row = pd.DataFrame([[self.training_date,self.model_type, population_type, sample_size,
                             recall, precision, f1, npv, log_loss, accuracy, balanced_accuracy, tp, tn, fp, fn ]], 
                           columns = ['training_date','Model_Type', 'population_type', 'sample_size',
                                      'recall', 'precision', 'f1', 'npv', 'logloss', 'accuracy', 'balanced_accuracy', 'tp', 'tn','fp', 'fn'])
    
    self.performance_metric_results = pd.concat([self.performance_metric_results, new_row], ignore_index = True)
    self.performance_metric_results.reset_index(drop = True, inplace = True)
    
    if population_type == 'Final':
      
      self._create_champion_model_performance_metrics(record = new_row)
        
    return
  
  def _create_champion_model_performance_metrics(self, record: pd.DataFrame) -> DataFrame:
    
    """
    Creates DataFrame Record Containing Model Metadata
    
    """
    
    record['model_path'] = self.model_path
    record['model_name'] = self.model_name
    record['Feature_Engineering_Path'] = self.feature_engineering_path
    
    record = record[['Feature_Engineering_Path','model_name', 'Model_Type','population_type','model_path', 'training_date', 'sample_size', 
                     'recall', 'precision', 'f1', 'npv', 'logloss', 'accuracy', 'balanced_accuracy', 'tp', 'tn','fp', 'fn']]
    
    self.champion_model_performance_metrics = record.copy()
    
    return self.champion_model_performance_metrics
  
  def _run_mlflow_experiements(self):
    
    """
    Evaluates ML Models & Tracks Artifacts Using MLFlow
    
    """
    
    experiment_full_path = os.path.join(self.mlflow_experiment_path, self.experiment_name)
    
    mlflow.create_experiment(experiment_full_path, self.mlflow_artifacts_path)
    mlflow.set_experiment(experiment_full_path)
    
    with mlflow.start_run() as run:

      self.ml_model = self.crossvalidation_pipeline.fit(self.train)

      training_predictions = self.ml_model.transform(self.train)
      test_predictions = self.ml_model.transform(self.test)
      
      self._evaluate_model(prediction_data = training_predictions, population_type = 'Training', model_run = 1, use_mlflow = 1)
      self._evaluate_model(prediction_data = test_predictions, population_type = 'Test', model_run = 1, use_mlflow = 1)
      
      mlflow.spark.log_model(spark_model = self.ml_model.bestModel, artifact_path = os.path.join(self.mlflow_artifacts_path, 'best-model'))
  
      mlflow.end_run()
    
    return
  
  def _run_experiments(self):
    
    """
    Evaluates ML Models Without MLFlow
    
    """
    
    self.ml_model = self.crossvalidation_pipeline.fit(self.train)

    training_predictions = self.ml_model.transform(self.train)
    test_predictions = self.ml_model.transform(self.test)
      
    self._evaluate_model(prediction_data = training_predictions, population_type = 'Training', model_run = 1, use_mlflow = 0)
    self._evaluate_model(prediction_data = test_predictions, population_type = 'Test', model_run = 1, use_mlflow = 0)
    
    return
  
  def _select_champion_model(self):
    
    """
    Selects Champion Model Based On Models' Performance On Test Set
    
    """
    
    self.best_parameters = [self.ml_model.bestModel.extractParamMap()]
  
    self.champion_crossvalidation_pipeline = CrossValidator(estimator = self.model,
                                                            estimatorParamMaps = self.best_parameters,
                                                            evaluator = self.model_evaluator,
                                                            numFolds = 2) 
    
    
    self.all_data = self.train.union(self.test)
    
    self.champion_model = self.champion_crossvalidation_pipeline.fit(self.all_data)
  
    
    if self.model_type=='xgboost': 
      self.champion_model = self.champion_model.bestModel
      
      self.champion_model.write() \
                   .overwrite() \
                   .save(self.model_path)
    else:
      self.champion_model.write() \
                         .overwrite() \
                         .save(self.model_path)

    predictions = self.champion_model.transform(self.all_data)
    
    if self.use_mlflow:
    
      self._evaluate_model(prediction_data = predictions, population_type = 'Final', model_run = 1, use_mlflow = 1)
      
    else:
      
      self._evaluate_model(prediction_data = predictions, population_type = 'Final', model_run = 1, use_mlflow = 0)      
    
    self._save_model_performance_metrics()
    self._update_champion_model_performance_metric_table()

      
    return

  def _save_model_performance_metrics(self):

    """
    Exports Model Performance Metrics
    
    """
  
    self.performance_metric_results = spark.createDataFrame(self.performance_metric_results)
    
    self.performance_metric_results.write \
                                   .format('delta') \
                                   .option('maxRecordsPerFile', 2000) \
                                   .option("overwriteSchema", "false") \
                                   .mode('append') \
                                   .saveAsTable(self.model_performance_metrics_path)

    return
  
  def _update_champion_model_performance_metric_table(self):
    
    """
    Updates DARQ Risk Model Champion Model Performance Metric Table
    """
    
    try:

      champion_darq_risk_models_performance_metrics = spark.table(self.champion_model_performance_metrics_path)
      champion_darq_risk_models_performance_metrics = champion_darq_risk_models_performance_metrics.toPandas()

      new_row = self.champion_model_performance_metrics

      champion_darq_risk_models_performance_metrics = pd.concat([champion_darq_risk_models_performance_metrics, new_row], ignore_index = True)
      
      champion_darq_risk_models_performance_metrics = spark.createDataFrame(self.champion_model_performance_metrics)
      
    except:
      
      champion_darq_risk_models_performance_metrics = spark.createDataFrame(self.champion_model_performance_metrics)

    champion_darq_risk_models_performance_metrics.write \
                                   .format('delta') \
                                   .option('maxRecordsPerFile', 2000) \
                                   .option("overwriteSchema", "false") \
                                   .mode('append') \
                                   .saveAsTable(self.champion_model_performance_metrics_path)
    
    return

  def _export_performance_metrics(self) -> DataFrame:
    
    """
    Returns Model Performance Metrics
    
    """
    
    return self.performance_metric_results
  
  def _export_champion_model_performance_metrics(self) -> DataFrame:
    
    """
    Returns Champion Model Performance Metrics
    
    """
    
    return self.champion_model_performance_metrics
  
  def _export_experiment_name(self) -> str:
    
    """
    Returns MLFlow Model Experiment Name
    
    """
    
    return self.experiment_name
  
  def _return_champion_model(self):
    
    """
    Returns Champion DARQ Risk Model
    
    """    

    return self.champion_model

# COMMAND ----------

  model_training = Train_Model(
                               model_type = model_type,
                               maximum_training_size = maximum_training_set_size,
                               performance_metric = model_performance_metric,
                               training_date = training_date,
                               use_mlflow = use_mlflow                               
                              )  
  
  print( '  [ SUCCESSFULLY INITIATED DARQ RISK CLASSIFICATION MODEL TRAINING ] ' )
  
  # Step 4.2 - Train DARQ Risk Model

  print( ' [ TRAINING DARQ RISK CLASSIFICATION MODEL ... ] ' )  

  model_training._train_model()

  print( '  [ SUCCESSFULLY TRAINED DARQ RISK CLASSIFICATION MODEL ] ' )   

  # Step 4.3 - Retreive Trained Champion DARQ Risk Model

  print( ' [ RETREIVE TRAINED CHAMPION DARQ RISK CLASSIFICATION MODEL ] ' )  

  champion_model = model_training._return_champion_model()
  champion_model_performance_metrics_df = model_training._export_champion_model_performance_metrics()

  #mlflow_experiment_name = model_training._export_experiment_name()

  print( '  [ SUCCESSFULLY RETREIVED TRAINED CHAMPION DE-OBLIGATION CLASSIFICATION MODEL ] ' ) 

# COMMAND ----------

champion_model_performance_metrics_df

# COMMAND ----------

spark.sql(r"SELECT * FROM fmddt_catalog.gao.predictive_labels_monthly_model_metrics_V1").display()
