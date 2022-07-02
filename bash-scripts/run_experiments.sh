#!/bin/bash

BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
FLINK_HOME=$BASEDIR/flink-1.14.2
DRIVER_HOME=$BASEDIR/driver
DATA_HOME=$BASEDIR/data
DRIVER_JAR_FILENAME="wordcount-driver-0.5.jar"

echo "Running from directory: $BASEDIR"

run_experiment()
{
	$FLINK_HOME/bin/stop-cluster.sh
	sleep 5
	CONF_FILE_SUFFIX=$6
	echo "Copying file $BASEDIR/conf/flink-conf_$CONF_FILE_SUFFIX.yaml to $FLINK_HOME/conf/flink-conf.yaml"
	cp $BASEDIR/conf/flink-conf_$CONF_FILE_SUFFIX.yaml $FLINK_HOME/conf/flink-conf.yaml
	$FLINK_HOME/bin/start-cluster.sh
	sleep 15
	CONFIG_STRING="$1,$2,$3,$4,$5"
	echo "Going to start driver program with config-string=$CONFIG_STRING"
	$FLINK_HOME/bin/flink run $DRIVER_HOME/$DRIVER_JAR_FILENAME -output file://$DRIVER_HOME/count_output_$CONF_FILE_SUFFIX \
		-config-string $CONFIG_STRING > $DRIVER_HOME/driver_out_$CONF_FILE_SUFFIX.log 2>&1 &
	echo "Started driver program. Going to sleep $1 minutes."
	sleep "$1"m
	echo "Driver program should be finished or finishing soon. Sleeping 30 seconds more before shutdown..."
	sleep 30
	echo "Going to shutdown cluster..."
	$FLINK_HOME/bin/stop-cluster.sh
	sleep 5
	echo "Going to get data for the last $(($1 + 2)) minutes."
	(cd $DATA_HOME; $DATA_HOME/dump_metrics.sh $(($1 + 2)) $CONF_FILE_SUFFIX)
}

echo "Running new default configuration"
run_experiment $1 $2 $3 $4 $5 newdefault
echo "Sleeping 0.5 minute(s)."
sleep 30

#echo "Running traffic-based configuration"
#run_experiment $1 $2 $3 traffic-based
#echo "Sleeping 5 minutes."
#sleep 5m

#echo "Running drl-based configuration"
#run_experiment $1 $2 $3 drl-based
#echo "Sleeping 5 minutes."
#sleep 5m

#echo "Running adaptive configuration"
#run_experiment $1 $2 $3 adaptive

echo "Done."
