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

import org.apache.flink.streaming.api.functions.source.FromSplittableIteratorFunction;
import org.apache.flink.util.SplittableIterator;
import org.astro.driver.util.UnboundedIterator;

import java.util.Iterator;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Provides the default data sets used for the WordCount example program.
 * The default data sets are used, if no parameters are given to the program.
 */
public class SplittableWordSource extends FromSplittableIteratorFunction<String> {

	public SplittableWordSource() {
		this(TimeUnit.MINUTES.toMillis(20), 4);
	}

	public SplittableWordSource(final long totalDuration,
								final int maxSplits) {
		super(new SplittableRateLimitedTextIterator(totalDuration, maxSplits));
	}

	private static class SplittableRateLimitedTextIterator extends SplittableIterator<String> {
		private static final long serialVersionUID = 1L;

		private UnboundedIterator[] innerArray;
		private final long totalDuration;
		private final AtomicInteger nextSplit = new AtomicInteger(-1);
		private int numberOfSplits;
		private final int maxSplits;
		private final long startTime;

		private SplittableRateLimitedTextIterator(final long totalDuration, final int maxSplits) {
			this.totalDuration = totalDuration;
			this.maxSplits = maxSplits;
			this.numberOfSplits = maxSplits;
			this.startTime = System.currentTimeMillis();
		}

		public boolean hasNext() {
			nextSplit.compareAndSet(numberOfSplits, -1);
			return this.innerArray[nextSplit.incrementAndGet()].hasNext();
		}

		public String next() {
			return this.innerArray[nextSplit.get()].next();
		}

		@Override
		public Iterator<String>[] split(int nSplits) {
			if (innerArray == null || numberOfSplits != nSplits) {
				innerArray = new UnboundedIterator[nSplits];
				for (int i = 0; i < nSplits; i++) {
					innerArray[i] = UnboundedIterator.unbounded(totalDuration, startTime);
				}
				numberOfSplits = nSplits;
			}
			return innerArray;
		}

		@Override
		public int getMaximumNumberOfSplits() {
			return maxSplits;
		}
	}
}
