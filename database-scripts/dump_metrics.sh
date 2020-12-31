#!/bin/bash

INFLUX_CMD='/home/lasantha/Software/influxdb-1.8.3-1/influx'

$INFLUX_CMD -database 'flink' -execute "SELECT * FROM taskmanager_System_CPU_Usage WHERE time > now() - $1m" \
-format csv > taskmanager_System_CPU_Usage_$2_`date -u +%Y_%m_%d`.csv
$INFLUX_CMD -database 'flink' -execute "SELECT * FROM jobmanager_System_CPU_Usage WHERE time > now() - $1m" \
-format csv > jobmanager_System_CPU_Usage_$2_`date -u +%Y_%m_%d`.csv

$INFLUX_CMD -database 'flink' -execute "SELECT operator_name, subtask_index, rate FROM taskmanager_job_task_operator_numRecordsOutPerSecond WHERE time > now() - $1m" \
-format csv > taskmanager_job_task_operator_numRecordsOutPerSecond_$2_`date -u +%Y_%m_%d`.csv
$INFLUX_CMD -database 'flink' -execute "SELECT operator_name, subtask_index, rate FROM taskmanager_job_task_operator_numRecordsInPerSecond WHERE time > now() - $1m" \
-format csv > taskmanager_job_task_operator_numRecordsInPerSecond_$2_`date -u +%Y_%m_%d`.csv
$INFLUX_CMD -database 'flink' -execute "SELECT edge_name, rate FROM taskmanager_job_task_edge_numRecordsProcessedPerSecond WHERE time > now() - $1m" \
-format csv > taskmanager_job_task_operator_numRecordsProcessedPerSecond_$2_`date -u +%Y_%m_%d`.csv
$INFLUX_CMD -database 'flink' -execute "SELECT operator_name, subtask_index, value FROM taskmanager_job_task_operator_currentCpuUsage WHERE time > now() - $1m" \
-format csv > taskmanager_job_task_operator_currentCpuUsage_$2_`date -u +%Y_%m_%d`.csv

for i in {0..11}
do
  QUERY="SELECT value FROM taskmanager_System_CPU_UsageCPU$i WHERE time > now() - $1m"
  FILENAME="jobmanager_System_CPU_UsageCPU"$i"_$2_`date -u +%Y_%m_%d`.csv"
  echo "Running query: $QUERY"
  $INFLUX_CMD -database 'flink' -execute "$QUERY" \
  -format csv > $FILENAME
done

$INFLUX_CMD -database 'flink-transitions' -execute "SELECT placementAction, cpuUsageMetrics, arrivalRate, throughput FROM state_snapshots WHERE time > now() - $1m" \
-format csv > state_snapshots_$2_`date -u +%Y_%m_%d`.csv
$INFLUX_CMD -database 'flink-transitions' -execute "SELECT action, newState, oldState, reward FROM state_transitions WHERE time > now() - $1m" \
-format csv > state_transitions_$2_`date -u +%Y_%m_%d`.csv
