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

import java.util.concurrent.TimeUnit;

/**
 * Skeleton code for the datastream walkthrough
 */
public class WordCountWithWatermarks {
    public static void main(String[] args) throws Exception {
        // Checking input parameters
        final MultipleParameterTool params = MultipleParameterTool.fromArgs(args);

        // set up the execution environment
        final StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        String outputPath = "word-count-output";
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

        DataStream<String> textStream = env
            .addSource(wordSource)
            .setParallelism(sourceParallelism)
            .name("sentence-stream")
            .assignTimestampsAndWatermarks(
                WatermarkStrategy
                    .<String>forMonotonousTimestamps()
                //.withTimestampAssigner()
                // .<String>forBoundedOutOfOrderness(Duration.ofSeconds(5))
                // .withTimestampAssigner((event, timestamp) -> event.f0)
            );


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
            .assignTimestampsAndWatermarks(
                WatermarkStrategy
                    .<WordCount>forBoundedOutOfOrderness(Duration.ofSeconds(20))
                // .withTimestampAssigner((event, timestamp) -> event.f0)
            )
            .name("word-count");

        countWords.addSink(fileSink)
            .setParallelism(sinkParallelism)
            .name("count-sink");

        env.enableCheckpointing(TimeUnit.MINUTES.toMillis(5));

        env.execute("Word Count");
    }

}