#!/bin/bash

DIST_UNPACKED_DIR=/home/m34ferna/flink-tests/build/dist-unpacked
FLINK_SRC_DIR=
FLINK_VER="2.12-1.14.2"
FLINK_DISTRO_VER="1.14.2"
TARGET_FLINK_DIR=/home/m34ferna/flink-tests/flink-schedule-modes-setup/flink-$FLINK_DISTRO_VER
#MVN_ADDITIONAL_ARGS="-Dscala-2.12 -DskipTests -Dcheckstyle.skip -Dspotless.apply.skip"
MVN_ADDITIONAL_ARGS="-Dscala-2.12 -DskipTests"

case "$1" in

1)	echo "Build flink-streaming-java, copying to 'dist-unpacked', copy the re-archived dist jar, and run..."
	mvn -f ~/src/flink-scheduler/flink-streaming-java/pom.xml clean install $MVN_ADDITIONAL_ARGS &&
		cp ~/src/flink-scheduler/flink-streaming-java/target/classes/org/apache/flink/streaming/* $DIST_UNPACKED_DIR/org/apache/flink/streaming/ -r &&
		rm $DIST_UNPACKED_DIR/flink-dist_$FLINK_VER.jar &&
		(cd $DIST_UNPACKED_DIR; jar cvf flink-dist_$FLINK_VER.jar ./*) &&
		cp $DIST_UNPACKED_DIR/flink-dist_$FLINK_VER.jar $TARGET_FLINK_DIR/lib/ # &&
		# $TARGET_FLINK_DIR/bin/start-cluster.sh
	;;

2)	echo "Build flink-runtime, flink-streaming-java, copying to 'dist-unpacked', copy the re-archived dist jar, and run..."
	mvn -f ~/src/flink-scheduler/flink-runtime/pom.xml clean install $MVN_ADDITIONAL_ARGS &&
		cp ~/src/flink-scheduler/flink-runtime/target/classes/org/apache/flink/runtime/* $DIST_UNPACKED_DIR/org/apache/flink/runtime/ -r &&
		mvn -f ~/src/flink-scheduler/flink-streaming-java/pom.xml clean install $MVN_ADDITIONAL_ARGS &&
		cp ~/src/flink-scheduler/flink-streaming-java/target/classes/org/apache/flink/streaming/* $DIST_UNPACKED_DIR/org/apache/flink/streaming/ -r &&
		rm $DIST_UNPACKED_DIR/flink-dist_$FLINK_VER.jar &&
		(cd $DIST_UNPACKED_DIR; jar cvf flink-dist_$FLINK_VER.jar ./*) &&
		cp $DIST_UNPACKED_DIR/flink-dist_$FLINK_VER.jar $TARGET_FLINK_DIR/lib/ # &&
		# $TARGET_FLINK_DIR/bin/start-cluster.sh
	;;

3)	echo "Build flink-runtime, flink-streaming-java, and core. Copying to 'dist-unpacked', copy the re-archived dist jar, and run..."
	mvn -f ~/src/flink-scheduler/flink-core/pom.xml clean install $MVN_ADDITIONAL_ARGS &&
		cp ~/src/flink-scheduler/flink-core/target/classes/org/apache/flink/* $DIST_UNPACKED_DIR/org/apache/flink/ -r &&
		mvn -f ~/src/flink-scheduler/flink-runtime/pom.xml clean install $MVN_ADDITIONAL_ARGS &&
		cp ~/src/flink-scheduler/flink-runtime/target/classes/org/apache/flink/runtime/* $DIST_UNPACKED_DIR/org/apache/flink/runtime/ -r &&
		mvn -f ~/src/flink-scheduler/flink-streaming-java/pom.xml clean install $MVN_ADDITIONAL_ARGS &&
		cp ~/src/flink-scheduler/flink-streaming-java/target/classes/org/apache/flink/streaming/* $DIST_UNPACKED_DIR/org/apache/flink/streaming/ -r &&
		rm $DIST_UNPACKED_DIR/flink-dist_$FLINK_VER.jar &&
		(cd $DIST_UNPACKED_DIR; jar cvf flink-dist_$FLINK_VER.jar ./*) &&
		cp $DIST_UNPACKED_DIR/flink-dist_$FLINK_VER.jar $TARGET_FLINK_DIR/lib/ # &&
		# $TARGET_FLINK_DIR/bin/start-cluster.sh
	;;

4)	echo "Build flink-core, flink-runtime, flink-streaming-java, and flink-connectors. Copying to 'dist-unpacked', copy the re-archived dist jar, and run..."
	mvn -f ~/src/flink-scheduler/flink-core/pom.xml clean install $MVN_ADDITIONAL_ARGS &&
		cp ~/src/flink-scheduler/flink-core/target/classes/org/apache/flink/* $DIST_UNPACKED_DIR/org/apache/flink/ -r &&
		mvn -f ~/src/flink-scheduler/flink-runtime/pom.xml clean install $MVN_ADDITIONAL_ARGS &&
		cp ~/src/flink-scheduler/flink-runtime/target/classes/org/apache/flink/runtime/* $DIST_UNPACKED_DIR/org/apache/flink/runtime/ -r &&
		mvn -f ~/src/flink-scheduler/flink-streaming-java/pom.xml clean install $MVN_ADDITIONAL_ARGS &&
		cp ~/src/flink-scheduler/flink-streaming-java/target/classes/org/apache/flink/streaming/* $DIST_UNPACKED_DIR/org/apache/flink/streaming/ -r &&
		mvn -f ~/src/flink-scheduler/flink-connectors/pom.xml clean install $MVN_ADDITIONAL_ARGS
		rm $DIST_UNPACKED_DIR/flink-dist_$FLINK_VER.jar &&
		(cd $DIST_UNPACKED_DIR; jar cvf flink-dist_$FLINK_VER.jar ./*) &&
		cp $DIST_UNPACKED_DIR/flink-dist_$FLINK_VER.jar $TARGET_FLINK_DIR/lib/
		# && $TARGET_FLINK_DIR/bin/start-cluster.sh
	;;


*)	echo "ERROR: Unrecognized option"
	;;
esac
