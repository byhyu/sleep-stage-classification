from pyspark.sql.functions import udf
import pyspark
from pyspark.sql import SparkSession
from pyspark import SparkConf
from pyspark.sql.types import MapType, StringType, IntegerType, StructType, StructField, ArrayType, FloatType
conf = SparkConf().setAppName('keras_spark_mnist').set('spark.sql.shuffle.partitions', '16')
# if args.master:
#     conf.setMaster(args.master)
# elif args.num_proc:
#     conf.setMaster('local[{}]'.format(args.num_proc))
spark = SparkSession.builder.config(conf=conf).getOrCreate()
print('done')

def process_edf(record_name):

    pass

udf_schema = StructType([
  StructField("patient_id", StringType(), True),
  StructField("comments", MapType(StringType(), StringType()), True),
  StructField("signals", MapType(StringType(), ArrayType(FloatType())), True),
])
extract_signals_udf = udf(extract_signals, udf_schema)