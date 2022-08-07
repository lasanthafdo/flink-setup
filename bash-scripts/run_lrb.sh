#!/bin/bash

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
FLINK_HOME=$BASE_DIR/flink-1.14.2
BENCHMARK_HOME="/home/m34ferna/src/flink-benchmarks"
DRIVER_HOME=$BENCHMARK_HOME/workload-processor-flink
WORKLOAD_GEN_HOME=$BENCHMARK_HOME/workload-generator
DATA_HOME=$BASE_DIR/data
DRIVER_JAR_FILENAME="workload-processor-flink-0.5.0.jar"
SETUP_FILE=$BENCHMARK_HOME/setup.yaml
EXPERIMENT_FILE=$BENCHMARK_HOME/benchmark/experiments/lrb_default.yaml

NUM_PARTITIONS=`yq e '."kafka.partitions"' $SETUP_FILE`

echo "Running from directory $BASE_DIR with $NUM_PARTITIONS Kafka partitions..."

run_experiment()
{
	$FLINK_HOME/bin/stop-cluster.sh
	sleep 5

	CONF_FILE_SUFFIX=$2
	echo "Copying file $BASE_DIR/conf/flink-conf_$CONF_FILE_SUFFIX.yaml to $FLINK_HOME/conf/flink-conf.yaml"
	cp $BASE_DIR/conf/flink-conf_$CONF_FILE_SUFFIX.yaml $FLINK_HOME/conf/flink-conf.yaml
	$FLINK_HOME/bin/start-cluster.sh
	$BENCHMARK_HOME/benchmark/kafka/bin/kafka-topics.sh --create --zookeeper localhost:2181 --replication-factor 1 --partitions $NUM_PARTITIONS --topic lrb-events-1
	sleep 10

	(cd $WORKLOAD_GEN_HOME; java -cp $WORKLOAD_GEN_HOME/target/workload-generator-0.5.0.jar \
		WorkloadGeneratorEntryPoint -s $SETUP_FILE -e $EXPERIMENT_FILE -r > workload_gen_$CONF_FILE_SUFFIX.log 2>&1 &)
	LOAD_PID=`ps -aef | grep WorkloadGeneratorEntryPoint | grep -v grep | awk '{print $2}'`
	echo "Started workload generator program with PID $LOAD_PID. Going to sleep $3 minutes"
	sleep "$3"m
	echo "Added data to Kafka topic. Going to kill workload generator program with PID $LOAD_PID"
	kill $LOAD_PID

	echo "Going to start driver program with setup file=$SETUP_FILE and experiment file=$EXPERIMENT_FILE"
	$FLINK_HOME/bin/flink run $DRIVER_HOME/target/$DRIVER_JAR_FILENAME --setup $SETUP_FILE \
		--experiment $EXPERIMENT_FILE > lrb_out_$CONF_FILE_SUFFIX.log 2>&1 &
	echo "Started driver program. Going to sleep $1 minutes..."
	sleep "$1"m


	echo "Going to shutdown cluster..."
	$FLINK_HOME/bin/stop-cluster.sh
	sleep 5

	$BENCHMARK_HOME/benchmark/kafka/bin/kafka-topics.sh --zookeeper localhost:2181 --delete --topic lrb-events-1 
	echo "Going to get data for the last $(($1 + 2)) minutes."
	(
		cd $DATA_HOME; 
		$DATA_HOME/dump_metrics.sh $(($1 + 2)) $CONF_FILE_SUFFIX;
		tar cvzf "$CONF_FILE_SUFFIX"_metrics_`date -u +%Y_%m_%d`.tar.gz *.csv && rm *.csv
	)
}

if [[ $# -lt 2 ]];
then
	echo "Invalid use: ./run_lrb_tembo.sh <TIME_IN_MINTUES> <EVENT_GEN_TIME_IN_MINUTES>"
else

	# echo "Running LRB default configuration"
	# FLINK_HOME=$BASE_DIR/vanilla-flink
	# run_experiment $1 lrb_default $2
	# echo "Sleeping 15 second(s)."
	# sleep 15

	echo "Running LRB adaptive configuration"
	FLINK_HOME=$BASE_DIR/flink-1.14.2
	run_experiment $1 lrb_adaptive $2
	echo "Sleeping 3 second(s)."
	sleep 3

	echo "Done."
fi
