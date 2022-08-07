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

import java.util.Objects;

/**
 * A simple alert event.
 */
@SuppressWarnings("unused")
public final class WordCountResult {

    private String word;
    private long count;

    private int windowCount;

    private long totalWindowWordCount;

    public WordCountResult(String word, long count, int windowCount, long totalWindowWordCount) {
        this.word = word;
        this.count = count;
        this.windowCount = windowCount;
        this.totalWindowWordCount = totalWindowWordCount;
    }

    public String getWord() {
        return word;
    }

    public void setWord(String word) {
        this.word = word;
    }

    public long getCount() {
        return count;
    }

    public void setCount(long count) {
        this.count = count;
    }

    public int getWindowCount() {
        return windowCount;
    }

    public void setWindowCount(int windowCount) {
        this.windowCount = windowCount;
    }

    public long getTotalWindowWordCount() {
        return totalWindowWordCount;
    }

    public void setTotalWindowWordCount(long totalWindowWordCount) {
        this.totalWindowWordCount = totalWindowWordCount;
    }

    @Override
    public String toString() {
        return "WordCountResult{" +
            "word='" + word + '\'' +
            ", count=" + count +
            ", windowCount=" + windowCount +
            ", totalWindowWordCount=" + totalWindowWordCount +
            '}';
    }
}
