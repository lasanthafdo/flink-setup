/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.astro.driver;

import org.apache.flink.api.common.serialization.SimpleStringEncoder;
import org.apache.flink.api.common.eventtime.*;
//import org.apache.flink.api.java.tuple.Tuple2;
import java.time.Duration;
import org.apache.flink.api.java.utils.MultipleParameterTool;
import org.apache.flink.core.fs.Path;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.sink.filesystem.StreamingFileSink;
import org.apache.flink.streaming.api.functions.sink.filesystem.rollingpolicies.DefaultRollingPolicy;
import org.astro.driver.entity.SplittableWordSource;
import org.astro.driver.entity.WordCount;
import org.astro.driver.entity.WordSplitter;

import java.time.Instant;
import java.util.Random;
import java.util.concurrent.TimeUnit;

/**
 * Skeleton code for the datastream walkthrough
 */
public class WordCountWithWatermarks {
    public static void main(String[] args) throws Exception {
        /* Obtain an execution environment for our streaming task */
        final StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

        /* Consume input parameters from command line */
        final MultipleParameterTool params = MultipleParameterTool.fromArgs(args);
        env.getConfig().setGlobalJobParameters(params);

        /* Parse command line arguments */
        String outputPath = "word-count-output";  // Path to the output folder
        if (params.has("output"))
            outputPath = params.get("output");

        int sourceParallelism = 3;
        int operatorParallelism = 3;
        int sinkParallelism = 3;
        SplittableWordSource wordSource;

        // make parameters available in the web interface
        env.getConfig().setGlobalJobParameters(params);

        if (params.has("output")) {
            outputPath = params.get("output");
        }

        if (params.has("config-string")) {
            String configString = params.get("config-string");
            String[] strParams = configString.split(",");
            System.out.println("Number of config parameters received: " + strParams.length);
            sourceParallelism = Integer.parseInt(strParams[1]);
            operatorParallelism = Integer.parseInt(strParams[2]);
            sinkParallelism = Integer.parseInt(strParams[3]);

            wordSource = new SplittableWordSource(
                TimeUnit.MINUTES.toMillis(Long.parseLong(strParams[0])),
                sourceParallelism
            );
        } else {
            wordSource = new SplittableWordSource();
        }

        /* Construct the data source */

        // 5 minutes = 300000 millis
        // To generate a number between `max` and `min`, use `random.nextInt(max - min) + min;`
        Random random = new Random();
        int randomShift = random.nextInt(600000) - 300000;
        WatermarkStrategy<String> watermarkStrategy = WatermarkStrategy
                .<String>forBoundedOutOfOrderness(Duration.ofSeconds(5))
                .withTimestampAssigner((event, timestamp) ->
                        Instant.now().toEpochMilli() + randomShift);

        DataStream<String> textStream = env
                .addSource(wordSource)
                .setParallelism(sourceParallelism)
                .assignTimestampsAndWatermarks(watermarkStrategy)
                .name("sentence-stream");

        DataStream<WordCount> splitWords = textStream
                .keyBy(String::toString)
                .process(new WordSplitter())
                .setParallelism(operatorParallelism)
                .name("word-splitter");

        final StreamingFileSink<WordCount> fileSink = StreamingFileSink
                .forRowFormat(new Path(outputPath), new SimpleStringEncoder<WordCount>("UTF-8"))
                .withRollingPolicy(
                        DefaultRollingPolicy.builder()
                                .withRolloverInterval(TimeUnit.MINUTES.toMillis(15))
                                .withInactivityInterval(TimeUnit.MINUTES.toMillis(10))
                                .withMaxPartSize(1024 * 1024 * 1024)
                                .build())
                .build();

        DataStream<WordCount> countWords = splitWords
                .keyBy(WordCount::getWord).sum("count")
                .setParallelism(operatorParallelism)
                .name("word-count");

        countWords.addSink(fileSink)
                .setParallelism(sinkParallelism)
                .name("count-sink");

        env.execute("Word Count");
    }
}