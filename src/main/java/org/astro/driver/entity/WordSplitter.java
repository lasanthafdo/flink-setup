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

package org.astro.driver.entity;

import org.astro.driver.entity.WordCount;
import org.apache.flink.streaming.api.functions.KeyedProcessFunction;
import org.apache.flink.util.Collector;

/**
 * Skeleton code for implementing a word splitter.
 */
public class WordSplitter extends KeyedProcessFunction<String, String, WordCount> {

    private static final long serialVersionUID = 1L;

    @Override
    public void processElement(
            String sentence,
            Context context,
            Collector<WordCount> collector) throws Exception {

        String[] tokens = sentence.toLowerCase().split("\\W+");

        // emit the pairs
        for (String token : tokens) {
            if (token.length() > 0) {
                WordCount alert = new WordCount();
                alert.setWord(token);
                alert.setCount(1);
                collector.collect(alert);
            }
        }
    }

}
