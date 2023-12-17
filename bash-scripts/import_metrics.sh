#!/bin/bash

if [[ $# -lt 2 ]]; then
	echo "Usage: import_metrics.sh <TARGET_DATE> <EXP_ID>"
fi

# Configure these according to local setup
REMOTE_USER="m34ferna"
ANALYSIS_SCRIPT_DIR=/home/m34ferna/src/flink-setup/analysis-scripts

MONOGRAPHY_SCRIPT=$ANALYSIS_SCRIPT_DIR/monography.py
STAT_METRICS_DIR=${1}-${2}
TARGET_DATE=`echo $1 | cut -d '-' -f2`
let PREV_DATE=TARGET_DATE-1
PREV_DATE=`printf '%02d' "$PREV_DATE"`
echo "Metrics directory: $STAT_METRICS_DIR, Target date: $TARGET_DATE, Previous date: $PREV_DATE"
mkdir -p $STAT_METRICS_DIR 
cd $STAT_METRICS_DIR

FORMAT_DATE=`echo ${1//-/ }`
METRICS_DATE=$(date --date="$(echo "$FORMAT_DATE 2023")" +"%Y_%m_%d")
PREV_METRICS_DATE=`echo ${METRICS_DATE}_${PREV_DATE} | cut -d '_' -f1,2,4`
echo "Metrics date: $METRICS_DATE , Previous metrics date: $PREV_METRICS_DATE"
scp "${REMOTE_USER}@tembo.cs.uwaterloo.ca:/home/${REMOTE_USER}/flink-scheduling/data/lrb_*_${METRICS_DATE}_${2}_*.tar.gz" .
cat *.tar.gz | tar xvzf - -i | grep .csv | awk -F'lrb' '{print $2}' | cut -d '_' -f2,4,5,6 > temp_output.txt
echo "Untar output captured."

sort -u temp_output.txt > unique_runs.txt
POLICIES=
MAX_ITER=1
PAR=
SRC_PAR=
DEF_ID_STR=
while read -r EXP_RUN_ID
do
	if [ -z "$EXP_RUN_ID" ]; then
		continue
	fi

	POLICY=`echo $EXP_RUN_ID | cut -d '_' -f1`
	PAR=`echo $EXP_RUN_ID | cut -d '_' -f2`
	SRC_PAR=`echo $EXP_RUN_ID | cut -d '_' -f3 | sed s/parts//`
	ITER=`echo $EXP_RUN_ID | cut -d '_' -f4 | sed s/iter//`
	if [[ $POLICIES != *"$POLICY"* ]]; then
		POLICIES=$POLICIES,lrb_$POLICY
		if [[ $POLICY == *"osdef"* ]];then
			DEF_ID_STR=lrb_$POLICY
		fi
	fi
	if [ "$ITER" -gt "$MAX_ITER" ]; then
		MAX_ITER=$ITER
	fi
done < unique_runs.txt
POLICIES="${POLICIES:1}"

echo "Found $POLICIES scheduling policies with parallelism $PAR and source parallelism $SRC_PAR for $MAX_ITER iterations"
echo "Default ID string: $DEF_ID_STR"
rename.ul -o $PREV_METRICS_DATE $METRICS_DATE *$PREV_METRICS_DATE.csv
cd ..
cd $ANALYSIS_SCRIPT_DIR
python3 $MONOGRAPHY_SCRIPT -p $PAR -sp $SRC_PAR -pol $POLICIES -i $MAX_ITER -def $DEF_ID_STR $STAT_METRICS_DIR $METRICS_DATE

PERF_METRICS_DIR=perf/$STAT_METRICS_DIR
echo "Perf metrics directory: $PERF_METRICS_DIR"
mkdir $PERF_METRICS_DIR
cd $PERF_METRICS_DIR

scp "${REMOTE_USER}@tembo.cs.uwaterloo.ca:/home/${REMOTE_USER}/flink-scheduling/data/perf/${STAT_METRICS_DIR}/*.txt" .
