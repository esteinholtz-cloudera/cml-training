import pyspark
from pyspark.sql import SparkSession

# Then call the `getOrCreate()` method of
# `SparkSession.builder` to start a Spark application.
# This example also gives the Spark application a name:

spark = SparkSession.builder \
  .appName('cml-training-pyspark') \
  .getOrCreate()

# Now you can use the `SparkSession` named `spark` to read
# data into Spark.


# ## Reading Data

# Read the flights dataset. This data is in CSV format
# and includes a header row. Spark can infer the schema
# automatically from the data:

flights = spark.read.csv('data/flights.csv', header=True, inferSchema=True)

# The result is a Spark DataFrame named `flights`.

# ## Using SQL Queries

# Instead of using Spark DataFrame methods, you can
# use a SQL query to achieve the same result.

# First you must create a temporary view with the
# DataFrame you want to query:

flights.createOrReplaceTempView('flights')

# Then you can use SQL to query the DataFrame:

spark.sql("""
  SELECT origin,
    COUNT(*) AS num_departures,
    AVG(dep_delay) AS avg_dep_delay
  FROM flights
  WHERE dest = 'SFO'
  GROUP BY origin
  ORDER BY avg_dep_delay""").toPandas()

