Spark Risk Monte Carlo
==============

A simple Spark application that calculates Value at Risk using the Monte Carlo method.

To make a jar:

    mvn package

To run from a gateway node in a CDH5.1+ cluster:

    spark-submit --class com.cloudera.datascience.montecarlorisk.MonteCarloRisk --master local \
      target/montecarlo-risk-0.0.1-SNAPSHOT.jar \
      <instruments file> <num trials> <parallellism> <factor means file> <factor covariances file>


e.g.

user@ubuntu:~/workspace/montecarlorisk$ /asidev/spark-latest/bin/spark-submit --class com.cloudera.datascience.montecarlorisk.MonteCarloRisk --master local target/montecarlo-risk-0.0.1-SNAPSHOT.jar /Users/yfang/workspace/montecarlorisk/data/instruments.csv 12000 8 /Users/yfang/workspace/montecarlorisk/data/means.csv /Users/yfang/workspace/montecarlorisk/data/covariances.csv




This will run the application in a single local process.  If the cluster is running a Spark standalone
cluster manager, you can replace "--master local" with "--master spark://`<master host>`:`<master port>`".

If the cluster is running YARN, you can replace "--master local" with "--master yarn".

