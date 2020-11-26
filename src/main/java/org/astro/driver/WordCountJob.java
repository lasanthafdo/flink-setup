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

import org.apache.flink.api.java.utils.MultipleParameterTool;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.astro.driver.entity.WordCount;
import org.astro.driver.entity.WordSource;
import org.astro.driver.entity.WordSplitter;

import java.util.concurrent.TimeUnit;

/**
 * Skeleton code for the datastream walkthrough
 */
public class WordCountJob {
    public static void main(String[] args) throws Exception {
        // Checking input parameters
        final MultipleParameterTool params = MultipleParameterTool.fromArgs(args);

        // set up the execution environment
        final StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        String outputPath = "word-count-output";
        // make parameters available in the web interface
        env.getConfig().setGlobalJobParameters(params);

        if (params.has("output")) {
            outputPath = params.get("output");
        }

        WordSource wordSource;
        if (params.has("config-string")) {
            String configString = params.get("config-string");
            String[] strParams = configString.split(",");
            System.out.println(strParams.length);
            wordSource = new WordSource(
                Integer.parseInt(strParams[0]),
                Integer.parseInt(strParams[1]),
                Integer.parseInt(strParams[2]),
                Long.parseLong(strParams[3]),
                TimeUnit.MINUTES.toMillis(Long.parseLong(strParams[4])),
                TimeUnit.MINUTES.toMillis(Long.parseLong(strParams[5])));
        } else {
            wordSource = new WordSource();
        }

        DataStream<String> textStream = env
            .addSource(wordSource)
            .name("stream");

        DataStream<WordCount> splitWords = textStream
            .keyBy(String::toString)
            .process(new WordSplitter())
            .name("word-splitter");

        DataStream<WordCount> countWords = splitWords
            .keyBy(WordCount::getWord).sum("count")
            .name("word-count");

        countWords
            .writeAsText(outputPath)
            .name("count-sink");

        env.execute("Word Count");
    }

}
