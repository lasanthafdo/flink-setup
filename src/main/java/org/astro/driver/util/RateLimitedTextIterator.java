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

package org.astro.driver.util;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

/**
 * An iterator of transaction events.
 */
public final class RateLimitedTextIterator implements Iterator<String>, Serializable {

    private static final long serialVersionUID = 1L;

    private final boolean bounded;
    private final long totalDuration;
    private final long startTime;
    private final int burstThreshold;
    private final long sleepTimeLower;
    private final long sleepTimeUpper;
    private final Random rand = new Random();

    private int index = 0;
    private int burstCount = 0;

    public static RateLimitedTextIterator bounded(final long duration, final long sleepThreshold, final int burstThreshold) {
        return new RateLimitedTextIterator(true, duration, sleepThreshold, burstThreshold);
    }

    public static RateLimitedTextIterator unbounded(final long duration, final long sleepThreshold, final int burstThreshold) {
        return new RateLimitedTextIterator(false, duration, sleepThreshold, burstThreshold);
    }

    private RateLimitedTextIterator(boolean bounded, final long duration, final long sleepThreshold, final int burstThreshold) {
        this.bounded = bounded;
        this.totalDuration = duration;
        this.burstThreshold = burstThreshold;

        long sleepMargin = (long) Math.ceil(sleepThreshold * 0.05);
        this.sleepTimeLower = sleepThreshold - sleepMargin;
        this.sleepTimeUpper = sleepThreshold + sleepMargin;
        this.startTime = System.currentTimeMillis();
    }

    @Override
    public boolean hasNext() {
        if (totalDuration < 0 || System.currentTimeMillis() - startTime < totalDuration) {
            if (index < data.size()) {
                return true;
            } else if (!bounded) {
                index = 0;
                return true;
            } else {
                return false;
            }
        } else {
            return false;
        }

    }

    @Override
    public String next() {
        if (burstCount++ >= burstThreshold) {
            long sleepTime = rand.longs(sleepTimeLower, (sleepTimeUpper + 1)).limit(1).findFirst().getAsLong();
            try {
                Thread.sleep(sleepTime);
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
            burstCount = 1;
        }

        return data.get(index++);
    }

    public static final List<String> data = Arrays.asList(
        "To be, or not to be,--that is the question:--",
        "Whether 'tis nobler in the mind to suffer",
        "The slings and arrows of outrageous fortune",
        "Or to take arms against a sea of troubles,",
        "And by opposing end them?--To die,--to sleep,--",
        "No more; and by a sleep to say we end",
        "The heartache, and the thousand natural shocks",
        "That flesh is heir to,--'tis a consummation",
        "Devoutly to be wish'd. To die,--to sleep;--",
        "To sleep! perchance to dream:--ay, there's the rub;",
        "For in that sleep of death what dreams may come,",
        "When we have shuffled off this mortal coil,",
        "Must give us pause: there's the respect",
        "That makes calamity of so long life;",
        "For who would bear the whips and scorns of time,",
        "The oppressor's wrong, the proud man's contumely,",
        "The pangs of despis'd love, the law's delay,",
        "The insolence of office, and the spurns",
        "That patient merit of the unworthy takes,",
        "When he himself might his quietus make",
        "With a bare bodkin? who would these fardels bear,",
        "To grunt and sweat under a weary life,",
        "But that the dread of something after death,--",
        "The undiscover'd country, from whose bourn",
        "No traveller returns,--puzzles the will,",
        "And makes us rather bear those ills we have",
        "Than fly to others that we know not of?",
        "Thus conscience does make cowards of us all;",
        "And thus the native hue of resolution",
        "Is sicklied o'er with the pale cast of thought;",
        "And enterprises of great pith and moment,",
        "With this regard, their currents turn awry,",
        "And lose the name of action.--Soft you now!",
        "The fair Ophelia!--Nymph, in thy orisons",
        "Be all my sins remember'd.");

}
