import pyspark.sql.functions as f
from pyspark.sql import Window, DataFrame
from pyspark.ml.stat import Correlation
from pyspark.sql.functions import col

import pandas as pd
import numpy as np
import re
import os
import json
from datetime import date
from functools import reduce
from functools import partial
import sys
import copy
from scipy.stats import pearsonr
from ast import literal_eval
import plotly.express as px
from scipy.stats import t
import math
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly
from pyspark.ml.feature import QuantileDiscretizer, VectorAssembler, Bucketizer
from pyspark.ml.stat import Correlation
from copy import deepcopy
from pyspark.sql.types import DateType

from pyspark.mllib.evaluation import MulticlassMetrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import itertools
from pyspark.ml.functions import vector_to_array
from pyspark.sql.types import StructType, DoubleType, ArrayType, StringType, DateType, Row 
from pyspark.ml.feature import OneHotEncoder, StringIndexer, StandardScaler, VectorAssembler, StringIndexerModel, CountVectorizerModel
from pyspark.ml import Pipeline, PipelineModel
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from pyspark.ml.feature import OneHotEncoder, StringIndexer, StandardScaler, VectorAssembler, Imputer, CountVectorizer
from pyspark.ml import Pipeline
from pyspark.sql.functions import kurtosis


##FEATURE IMPORTANCE 
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, CrossValidatorModel
import json



def get_catagorical_names(pipeline_model):
  """
  Assumes all catagorical features are held within a StringIndexer or CountVectorizer
  """
  col_names = []
  for component in pipeline_model.stages:
  #get StringIndexer column names
    if isinstance(component, StringIndexerModel):
      feature_name = component.getInputCol()
      col_names += [feature_name +'_' + label for label in component.labels]
      
    
  #get CountVectorizer Column Names
    if isinstance(component, CountVectorizerModel):
      feature_name = component.getInputCol()
      col_names += [feature_name + '_' + label for label in component.vocabulary]
  return col_names


# Helper functon to select features to scale given their skew 
def select_features_to_scale(df, lower_skew=-2,upper_skew=2, continuous_cols=[], drop_cols=['']):
  # Check for kurtosis 
  scoring = df.agg(*(f.kurtosis(colm).alias(colm) for colm in continuous_cols)).toPandas()
  # melt dataframe to loop through and check for scaling 
  scoring = pd.melt(scoring,var_name='col_names',value_name='kurtosis')
  # Loop through 'feature_list' to select features based on Kurtisus / Skew
  selected_features = [col_name for col_name,k in zip(scoring['col_names'],scoring['kurtosis']) if (k<-2) or (k>2)]
  
  return selected_features


def feature_engineering(data: DataFrame, cat_features : list, num_features: list,count_vec_features : list ):
    """Contains scaler, feature vectorizer, one hot encoding. Creates a pipeline for reproducibility"""
    column_names = []
    stages = []
    #Scaled Features
    
    features_to_scale = select_features_to_scale(df=data,continuous_cols=num_features, drop_cols=[''] )
    print(f'>>> scaling: {features_to_scale}')

    unscaled_assembler = VectorAssembler(inputCols= features_to_scale, outputCol='to_scale_vec')
    scaler = StandardScaler(inputCol='to_scale_vec',outputCol='scaled_features')
    stages += [unscaled_assembler, scaler]

    #Unscaled
    
    #create list of numeric features that are not being scaled 
    num_unscaled_diff_list = list(set(num_features)-set(features_to_scale))
    
    unscaled_features = num_unscaled_diff_list
    assembler_1 = VectorAssembler(inputCols=unscaled_features, outputCol="unscaled_vec")
    stages += [assembler_1]
    #Catagorical
    
    cat_names = []
    for features in cat_features:
      #- Index Categorical Feautures
      string_indexer = StringIndexer(inputCol=features, outputCol= features+"_index",handleInvalid="keep")
      # One hot encode categorical features 
      encoder = OneHotEncoder(inputCols=[string_indexer.getOutputCol()],
                                    outputCols=[features+"_class_vec"])
      stages += [string_indexer,encoder]
    for features in count_vec_features: 
      #string_indexer2 = StringIndexer(inputCol=features, outputCol= features+"_count_vec_index")

      cv = CountVectorizer(inputCol=features,outputCol=features+"_count_class_vec",minDF=0.01)
      stages += [cv]
    
    cat_names = [feature+"_class_vec" for feature in cat_features] + [feature+"_count_class_vec" for feature in count_vec_features]
    assembler_2 = VectorAssembler(inputCols=cat_names, outputCol="cat_vec")
    stages += [assembler_2]
    
    # Combine Vectors

    # Assemble final Training data of scaled, numeric, and categorical  features 
    assembler_final = VectorAssembler(inputCols=['scaled_features','unscaled_vec','cat_vec'],outputCol='features')
    stages += [assembler_final]
    
    # Set Pipeline 
    pipeline = Pipeline(stages= stages)
    print('>>> fitting pipeline')
    # Fit pipeline to Data
    pipeline_model = pipeline.fit(data)
    
    # get column names
    column_names = features_to_scale + unscaled_features + get_catagorical_names(pipeline_model)
    
    # transform data using fitted pipeline
    df_transform = pipeline_model.transform(data)
    return (pipeline_model, df_transform, column_names)


class RandomGridBuilder(): 
  '''Grid builder for random search. Sets up grids for use in CrossValidator in Spark using values randomly sampled from user-provided distributions.
  Distributions should be provided as lambda functions, so that the numbers are generated at call time.
  
  Parameters:
    num_models: Integer (Python) - number of models to generate hyperparameters for
    seed: Integer (Python) - seed (optional, default is None)
    
  Returns:
    param_map: list of parameter maps to use in cross validation.
    
  Example usage:
    from pyspark.ml.classification import LogisticRegression
    lr = LogisticRegression()
    paramGrid = RandomGridBuilder(2)\
               .addDistr(lr.regParam, lambda: np.random.rand()) \
               .addDistr(lr.maxIter, lambda : np.random.randint(10))\
               .build()
               
    Returns similar output as Spark ML class ParamGridBuilder and can be used in its place. The above paramGrid provides random hyperparameters for 2 models.
    '''
  
  def __init__(self, num_models, seed=None):
    self._param_grid = {}
    self.num_models = num_models
    self.seed = seed
    
  def addDistr(self, param, distr_generator):
    '''Add distribution based on dictionary generated by function passed to addDistr.'''
    
    if 'pyspark.ml.param.Param' in str(type(param)):
      self._param_grid[param] = distr_generator
    else:
      raise TypeError('param must be an instance of Param')

    return self
  
  def build(self):    
    param_map = []
    for n in range(self.num_models):
      if self.seed:
        # Set seeds for both numpy and random in case either is used for the random distribution
        np.random.seed(self.seed + n)
        random.seed(self.seed + n)
      param_dict = {}
      for param, distr in self._param_grid.items():
        param_dict[param] = distr()
      param_map.append(param_dict)
    
    return param_map






