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

import org.apache.flink.api.common.functions.CoGroupFunction;
import org.apache.flink.api.common.serialization.SimpleStringEncoder;
import org.apache.flink.api.java.utils.MultipleParameterTool;
import org.apache.flink.core.fs.Path;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.SingleOutputStreamOperator;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.sink.SinkFunction;
import org.apache.flink.streaming.api.functions.sink.filesystem.StreamingFileSink;
import org.apache.flink.streaming.api.functions.sink.filesystem.rollingpolicies.DefaultRollingPolicy;
import org.apache.flink.streaming.api.windowing.assigners.TumblingEventTimeWindows;
import org.apache.flink.streaming.api.windowing.assigners.TumblingProcessingTimeWindows;
import org.apache.flink.streaming.api.windowing.time.Time;
import org.apache.flink.util.Collector;
import org.astro.driver.entity.SplittableWordSource;
import org.astro.driver.entity.WordCount;
import org.astro.driver.entity.WordCountResult;
import org.astro.driver.entity.WordSplitter;

import java.util.concurrent.TimeUnit;

/**
 * Skeleton code for the datastream walkthrough
 */
public class AdvancedWordCountJob {
	public static void main(String[] args) throws Exception {
		// Checking input parameters
		final MultipleParameterTool params = MultipleParameterTool.fromArgs(args);

		// set up the execution environment
		final StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
		SplittableWordSource wordSource;
		// make parameters available in the web interface
		env.getConfig().setGlobalJobParameters(params);

		if (params.has("config-string")) {
			String configString = params.get("config-string");
			String[] strParams = configString.split(",");
			System.out.println("Number of config parameters received: " + strParams.length);
			wordSource = new SplittableWordSource(
				TimeUnit.MINUTES.toMillis(Long.parseLong(strParams[0])),
				1
			);
		} else {
			wordSource = new SplittableWordSource();
		}

		DataStream<String> textStream = env
			.addSource(wordSource)
			.name("sentence-stream");

		DataStream<WordCount> splitWords = textStream
			.keyBy(String::toString)
			.process(new WordSplitter())
			.name("word-splitter");

		DataStream<WordCount> countWords = splitWords
			.keyBy(WordCount::getWord).sum("count")
			.name("word-count");

		DataStream<WordCount> windowedCountWords =
			splitWords.keyBy(WordCount::getWord).window(TumblingProcessingTimeWindows.of(Time.seconds(5))).sum("count").name("windowed-count");

		countWords.coGroup(windowedCountWords).where(WordCount::getWord).equalTo(WordCount::getWord).window(TumblingProcessingTimeWindows.of(Time.seconds(10))).apply(
			new CoGroupFunction<WordCount, WordCount, WordCountResult>() {
				@Override
				public void coGroup(Iterable<WordCount> first, Iterable<WordCount> second,
									Collector<WordCountResult> out) throws Exception {
					WordCount noWinHead = first.iterator().next();
					int windowCount = 0;
					long winWordCount = 0;
					for (WordCount winWord : second) {
						windowCount++;
						winWordCount += winWord.getCount();
					}
					out.collect(new WordCountResult(noWinHead.getWord(), noWinHead.getCount(), windowCount, winWordCount));
				}
			}
		).addSink(new SinkFunction<WordCountResult>() {
				@Override
				public void invoke(WordCountResult value, Context context) throws Exception {
					//do nothing
				}
			})
			.name("count-sink");

		// env.enableCheckpointing(TimeUnit.MINUTES.toMillis(5));

		env.execute("Advanced Word Count");
	}

}
