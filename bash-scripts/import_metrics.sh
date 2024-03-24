#!/bin/bash

if [[ $# -lt 3 ]]; then
	echo "Usage: import_metrics.sh <TARGET_DATE> <EXP_ID> <PREFIX_FILTER>"
	exit 0
fi

ANALYSIS_SCRIPT_DIR=/home/m34ferna/src/flink-setup/analysis-scripts
MONOGRAPHY_SCRIPT=$ANALYSIS_SCRIPT_DIR/monography_ysb.py
STAT_METRICS_DIR=${1}-${2}
PREFIX_FILTER=$3
EXP_HOST=`echo ${PREFIX_FILTER} | cut -d '_' -f1`
TARGET_DATE=`echo $1 | cut -d '-' -f2`
let PREV_DATE=TARGET_DATE-1
PREV_DATE=`printf '%02d' "$PREV_DATE"`
echo "Metrics directory: $STAT_METRICS_DIR, Target date: $TARGET_DATE, Previous date: $PREV_DATE"
mkdir -p $STAT_METRICS_DIR || exit 1
cd $STAT_METRICS_DIR

FORMAT_DATE=`echo ${1//-/ }`
METRICS_DATE=$(date --date="$(echo "$FORMAT_DATE 2024")" +"%Y_%m_%d")
PREV_METRICS_DATE=`echo ${METRICS_DATE}_${PREV_DATE} | cut -d '_' -f1,2,4`
echo "Metrics date: $METRICS_DATE , Previous metrics date: $PREV_METRICS_DATE"
scp "m34ferna@tembo.cs.uwaterloo.ca:/home/m34ferna/flink-scheduling/data/${PREFIX_FILTER}_*_${METRICS_DATE}_${2}_*.tar.gz" . || exit 1
cat *.tar.gz | tar xvzf - -i | grep .csv | awk -F"${PREFIX_FILTER}" '{print $2}' | cut -d '_' -f2,3,4,5,6 > temp_output.txt
echo "Untar output captured."

sort -u temp_output.txt > unique_runs.txt
POLICIES=
MAX_ITER=1
SCHED_PERIOD=
PAR=
SRC_PAR=
DEF_ID_STR=
while read -r EXP_RUN_ID
do
	if [ -z "$EXP_RUN_ID" ]; then
		continue
	fi

	POLICY=`echo $EXP_RUN_ID | cut -d '_' -f1`
	SCHED_PERIOD=`echo $EXP_RUN_ID | cut -d '_' -f2 | sed s/ms//` 
	PAR=`echo $EXP_RUN_ID | cut -d '_' -f3`
	SRC_PAR=`echo $EXP_RUN_ID | cut -d '_' -f4 | sed s/parts//`
	ITER=`echo $EXP_RUN_ID | cut -d '_' -f5 | sed s/iter//`
	if [[ $POLICIES != *"$POLICY"* ]]; then
		POLICIES=$POLICIES,ysb_$POLICY
		if [[ $POLICY == *"osdef"* ]];then
			DEF_ID_STR=ysb_$POLICY
		fi
	fi
	if [ "$ITER" -gt "$MAX_ITER" ]; then
		MAX_ITER=$ITER
	fi
done < unique_runs.txt
POLICIES="${POLICIES:1}"

if [ -z "$DEF_ID_STR" ]; then
	DEF_ID_STR=ysb_$POLICY
fi

echo "Found $POLICIES policies with par=$PAR and src_par=$SRC_PAR for $MAX_ITER iterations which ran on $EXP_HOST"
echo "Default ID string: $DEF_ID_STR"
rename "s/$PREV_METRICS_DATE/$METRICS_DATE/" *$PREV_METRICS_DATE.csv
cd ..
cd $ANALYSIS_SCRIPT_DIR
SCRIPT_COMMAND="python3 $MONOGRAPHY_SCRIPT -p $PAR -sp $SRC_PAR -pol $POLICIES -i $MAX_ITER -scp $SCHED_PERIOD -def $DEF_ID_STR --host $EXP_HOST $STAT_METRICS_DIR $METRICS_DATE"
echo "Executing the following command:"
echo $SCRIPT_COMMAND
$SCRIPT_COMMAND
# python3 $MONOGRAPHY_SCRIPT -p $PAR -sp $SRC_PAR -pol $POLICIES -i $MAX_ITER -scp $SCHED_PERIOD -def $DEF_ID_STR --host $EXP_HOST $STAT_METRICS_DIR $METRICS_DATE

PERF_DIR_EXISTS=`ssh m34ferna@tembo.cs.uwaterloo.ca test -d /home/m34ferna/flink-scheduling/data/perf/${STAT_METRICS_DIR} && echo "true"`
if [ -n "$PERF_DIR_EXISTS" ]; then
	PERF_METRICS_DIR=perf/$STAT_METRICS_DIR
	echo "Perf metrics directory: $PERF_METRICS_DIR"
	mkdir $PERF_METRICS_DIR
	cd $PERF_METRICS_DIR

	scp "m34ferna@tembo.cs.uwaterloo.ca:/home/m34ferna/flink-scheduling/data/perf/${STAT_METRICS_DIR}/*.txt" .
fi
