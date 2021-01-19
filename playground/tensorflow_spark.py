#%%
# Spark Session, Pipeline, Functions, and Metrics
#%%
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
# from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, StandardScaler, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.functions import rand
from pyspark.mllib.evaluation import MulticlassMetrics

# Keras / Deep Learning
from tensorflow.keras.models import Sequential
# from keras.models import Sequential
# from keras.layers.core import Dense, Dropout, Activation
# from keras import optimizers, regularizers
# from keras.optimizers import Adam

# Elephas for Deep Learning on Spark
from elephas.ml_model import ElephasEstimator
#%%
