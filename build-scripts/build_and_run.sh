#!/bin/bash

case "$1" in

1)	echo "Build only flink-runtime, copying to 'dist-unpacked', copy the re-archived dist jar, and run..."
	mvn -f ~/src/flink/flink-runtime/pom.xml clean install -Dcheckstyle.skip -DskipTests &&
		cp ~/src/flink/flink-core/target/classes/org/apache/flink/configuration/* ~/flink-tests/builds/jar-patching/dist-unpacked/org/apache/flink/configuration/ -r &&
		cp ~/src/flink/flink-streaming-java/target/classes/org/apache/flink/streaming/* ~/flink-tests/builds/jar-patching/dist-unpacked/org/apache/flink/streaming/ -r &&
		cp ~/src/flink/flink-runtime/target/classes/org/apache/flink/runtime/* ~/flink-tests/builds/jar-patching/dist-unpacked/org/apache/flink/runtime/ -r &&
		rm ~/flink-tests/builds/jar-patching/dist-unpacked/flink-dist_2.11-1.11.2-SNAPSHOT.jar &&
		(cd ~/flink-tests/builds/jar-patching/dist-unpacked; jar cvf flink-dist_2.11-1.11.2-SNAPSHOT.jar ./*) &&
		cp ~/flink-tests/builds/jar-patching/dist-unpacked/flink-dist_2.11-1.11.2-SNAPSHOT.jar flink-1/flink-1.11.2-SNAPSHOT/lib/ &&
		cp ~/flink-tests/builds/jar-patching/dist-unpacked/flink-dist_2.11-1.11.2-SNAPSHOT.jar flink-2/flink-1.11.2-SNAPSHOT/lib/ &&
		flink-1/flink-1.11.2-SNAPSHOT/bin/start-cluster.sh && flink-2/flink-1.11.2-SNAPSHOT/bin/taskmanager.sh start
	;;

2)	echo "Build full-distribution, copying to 'dist-unpacked', extract, copy again, and run..."
	mvn -f ~/src/flink/pom.xml clean install -Dcheckstyle.skip -DskipTests &&
		rm -r ~/flink-tests/builds/jar-patching/dist-unpacked/* &&
		cp ~/src/flink/flink-dist/target/flink-dist_2.11-1.11.2-SNAPSHOT.jar ~/flink-tests/builds/jar-patching/dist-unpacked/ -r &&
		(cd ~/flink-tests/builds/jar-patching/dist-unpacked; jar xvf flink-dist_2.11-1.11.2-SNAPSHOT.jar) &&
		cp ~/src/flink/flink-dist/target/flink-dist_2.11-1.11.2-SNAPSHOT.jar flink-1/flink-1.11.2-SNAPSHOT/lib/ &&
		cp ~/src/flink/flink-dist/target/flink-dist_2.11-1.11.2-SNAPSHOT.jar flink-2/flink-1.11.2-SNAPSHOT/lib/ &&
		flink-1/flink-1.11.2-SNAPSHOT/bin/start-cluster.sh && flink-2/flink-1.11.2-SNAPSHOT/bin/taskmanager.sh start
	;;

3)	echo "Build full distribution, copy, run, and show logs..."
	mvn -f ~/src/flink/pom.xml clean install -Dcheckstyle.skip -DskipTests &&
		cp ~/src/flink/flink-dist/target/flink-dist_2.11-1.11.2-SNAPSHOT.jar ~/Software/flink-1.11.2-SNAPSHOT/lib/ &&
		bin/start-cluster.sh &&
		tail -f log/flink-lasantha-standalonesession-0-lasantha-Lenovo-Legion-Y530-15ICH.log
	;;

*)	echo "ERROR: Unrecognized option"
	;;
esac
