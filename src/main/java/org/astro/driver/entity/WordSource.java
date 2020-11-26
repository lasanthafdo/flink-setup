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

import org.apache.flink.streaming.api.functions.source.FromIteratorFunction;
import org.astro.driver.util.TextIterator;

import java.io.Serializable;
import java.util.Iterator;
import java.util.concurrent.TimeUnit;

/**
 * Provides the default data sets used for the WordCount example program.
 * The default data sets are used, if no parameters are given to the program.
 */
public class WordSource extends FromIteratorFunction<String> {

    public WordSource() {
        super(new RateLimitedIterator<>(TextIterator.unbounded(), 200, 250000, 250000,
            200, TimeUnit.MINUTES.toMillis(5), TimeUnit.MINUTES.toMillis(20)));
    }

    public WordSource(final int normalBurstThreshold, final int burstLower,
                      final int burstStepSize, final long sleepTimeUpper,
                      final long epochDuration, final long totalDuration) {
        super(new RateLimitedIterator<>(TextIterator.unbounded(), normalBurstThreshold, burstLower,
            burstStepSize, sleepTimeUpper, epochDuration, totalDuration));
    }

    private static class RateLimitedIterator<T> implements Iterator<T>, Serializable {
        private static final long serialVersionUID = 1L;

        private final long startTime = System.currentTimeMillis();
        private final Iterator<T> inner;
        private long lastEpochTime;
        private int burstCount = 0;
        private int normalBurstCount = 0;
        private int burstThreshold;
        private long sleepTime;

        private final int normalBurstThreshold;
        private final long sleepTimeUpper;
        private final long totalDuration;
        private final long epochDuration;
        private final int burstStepSize;

        private RateLimitedIterator(Iterator<T> inner, final int normalBurstThreshold, final int burstLower,
                                    final int burstStepSize, final long sleepTimeUpper,
                                    final long epochDuration, final long totalDuration) {
            this.inner = inner;
            this.sleepTimeUpper = sleepTimeUpper;
            this.totalDuration = totalDuration;
            this.epochDuration = epochDuration;
            this.burstStepSize = burstStepSize;
            this.normalBurstThreshold = normalBurstThreshold;

            this.burstThreshold = burstLower;
            this.sleepTime = sleepTimeUpper;
            this.lastEpochTime = startTime;
        }

        public boolean hasNext() {
            if (System.currentTimeMillis() - startTime < totalDuration) {
                return this.inner.hasNext();
            } else {
                return false;
            }
        }

        public T next() {
            if ((System.currentTimeMillis() - lastEpochTime) < epochDuration) {
                if (normalBurstCount >= normalBurstThreshold) {
                    try {
                        Thread.sleep(sleepTime);
                        sleepTime -= 10;
                        if (sleepTime <= 0) {
                            sleepTime = sleepTimeUpper;
                        }
                    } catch (InterruptedException e) {
                        throw new RuntimeException(e);
                    }
                    normalBurstCount = 0;
                } else {
                    normalBurstCount++;
                }
            } else {
                if (burstCount < burstThreshold) {
                    burstCount++;
                } else {
                    lastEpochTime = System.currentTimeMillis();
                    burstCount = 0;
                    burstThreshold += burstStepSize;
                }
            }

            return this.inner.next();
        }
    }
}
