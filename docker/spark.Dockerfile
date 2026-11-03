FROM python:3.11-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-17-jre-headless procps \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    KAFKA_CONNECTOR=org.apache.spark:spark-sql-kafka-0-10_2.13:4.0.1 \
    IVY_DIR=/opt/ivy

RUN pip install pyspark==4.0.1

# Resolve the Kafka connector at build time so containers start without hitting Maven.
RUN python -c "import os; from pyspark.sql import SparkSession; \
SparkSession.builder.master('local[1]') \
.config('spark.jars.packages', os.environ['KAFKA_CONNECTOR']) \
.config('spark.jars.ivy', os.environ['IVY_DIR']).getOrCreate().stop()"

WORKDIR /app
COPY src/streamsentinel/streaming/spark_job.py /app/spark_job.py

CMD spark-submit --master "local[2]" \
    --conf spark.jars.ivy=${IVY_DIR} \
    --packages ${KAFKA_CONNECTOR} \
    /app/spark_job.py
