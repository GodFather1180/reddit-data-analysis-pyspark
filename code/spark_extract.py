"""
Spark job: pull the columns we need out of the course Reddit corpus and
write them back out as Parquet, filtered to a handful of years.

Run on the cluster:
    spark-submit spark_extract.py <output-dir>

The corpus is Hive-partitioned by year/month. Two things matter for making
this fast on a small teaching cluster:

1. We glob directly for `year={y1,y2,...}` instead of reading the whole
   root path and filtering afterwards — reading the root forces Spark to
   list every year=YYYY directory (2006-2025) before pruning.
2. We pass an explicit schema instead of letting `spark.read.json` infer
   one — schema inference means sampling every record across every file
   under the read path first, which dominates runtime on a corpus this
   size regardless of how narrow the glob is.

`created_utc` comes in as a STRING in this corpus (not a Long like you
might expect), so it's cast explicitly after reading.
"""
import sys

from pyspark.sql import SparkSession, functions as F, types as T

# Set this before running. Check what's actually on the cluster first:
#   hdfs dfs -ls /courses/datasets/reddit_submissions_repartitioned/
YEARS = [2022, 2023]

SUBMISSIONS_PATH = '/courses/datasets/reddit_submissions_repartitioned/'
COMMENTS_PATH = '/courses/datasets/reddit_comments_repartitioned/'

# Only the fields we use — an explicit schema means Spark parses just
# these out of each JSON record instead of inferring types from the ~40
# fields actually present.
SUBMISSION_SCHEMA = T.StructType([
    T.StructField('id', T.StringType()),
    T.StructField('subreddit', T.StringType()),
    T.StructField('author', T.StringType()),
    T.StructField('created_utc', T.StringType()),
    T.StructField('score', T.LongType()),
    T.StructField('num_comments', T.LongType()),
    T.StructField('title', T.StringType()),
    T.StructField('is_self', T.BooleanType()),
    T.StructField('stickied', T.BooleanType()),
    T.StructField('over_18', T.BooleanType()),
])

COMMENT_SCHEMA = T.StructType([
    T.StructField('id', T.StringType()),
    T.StructField('subreddit', T.StringType()),
    T.StructField('author', T.StringType()),
    T.StructField('created_utc', T.StringType()),
    T.StructField('score', T.LongType()),
])


def year_glob(base_path):
    years = '{' + ','.join(str(y) for y in YEARS) + '}'
    return f'{base_path}year={years}'


def extract(spark, path, schema, output_dir):
    df = (
        spark.read.option('basePath', path).schema(schema).json(year_glob(path))
        .withColumn('created_utc', F.col('created_utc').cast(T.LongType()))
        .coalesce(20)
    )
    df.write.mode('overwrite').option('compression', 'snappy').parquet(output_dir)


def main(output_dir):
    spark = SparkSession.builder.appName('reddit-extract').getOrCreate()
    spark.sparkContext.setLogLevel('WARN')

    extract(spark, SUBMISSIONS_PATH, SUBMISSION_SCHEMA, output_dir + '/submissions')
    extract(spark, COMMENTS_PATH, COMMENT_SCHEMA, output_dir + '/comments')

    spark.stop()


if __name__ == '__main__':
    main(sys.argv[1])
