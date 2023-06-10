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

import com.sun.jna.LastErrorException;
import com.sun.jna.Library;
import com.sun.jna.Native;
import com.sun.jna.NativeLong;
import com.sun.jna.Structure;

import java.lang.reflect.Field;
import java.util.BitSet;
import java.util.Collections;
import java.util.List;

/**
 * System information using syscalls.
 */
public class CpuAffinity {

    /*
     *  Some of the functionality is based on code from OpenHFT Java Affinity library
     *	at https://github.com/OpenHFT/Java-Thread-Affinity
     *
     * */
    public static class cpu_set_t extends Structure {
        static final int __CPU_SETSIZE = 1024;
        static final int __NCPUBITS = 8 * NativeLong.SIZE;
        static final int SIZE_OF_CPU_SET_T = (__CPU_SETSIZE / __NCPUBITS) * NativeLong.SIZE;
        static List<String> FIELD_ORDER = Collections.singletonList("__bits");
        public NativeLong[] __bits = new NativeLong[__CPU_SETSIZE / __NCPUBITS];

        public cpu_set_t() {
            for (int i = 0; i < __bits.length; i++) {
                __bits[i] = new NativeLong(0);
            }
        }

        @Override
        protected List<String> getFieldOrder() {
            return FIELD_ORDER;
        }
    }

    private interface CStdLib extends Library {
        int sched_getcpu() throws LastErrorException;

        int sched_setaffinity(final int pid, final int cpu_set_size, cpu_set_t cpu_mask)
            throws LastErrorException;

        int sched_getaffinity(final int pid, final int cpu_set_size, cpu_set_t cpu_mask)
            throws LastErrorException;

        int gettid() throws LastErrorException;
    }

    private final CStdLib cStdLib;

    private static CpuAffinity instance = new CpuAffinity();

    private CpuAffinity() {
        cStdLib = Native.loadLibrary("c", CStdLib.class);
    }

    public BitSet getCpuAffinity() {
        final cpu_set_t cpu_mask = new cpu_set_t();
        final int size = cpu_set_t.SIZE_OF_CPU_SET_T;
        int result = cStdLib.sched_getaffinity(0, size, cpu_mask);
        if (result == 0) {
            BitSet affinity = new BitSet(cpu_set_t.SIZE_OF_CPU_SET_T);
            int i = 0;
            for (NativeLong nativeLong : cpu_mask.__bits) {
                for (int j = 0; j < Long.SIZE; j++) {
                    affinity.set(i++, ((nativeLong.longValue() >>> j) & 1) != 0);
                }
            }
            return affinity;
        } else {
            throw new RuntimeException("Could not get the CPU affinity for this platform");
        }
    }

    public int setCpuAffinity(BitSet affinity) {
        cpu_set_t cpu_mask = new cpu_set_t();
        int size = cpu_set_t.SIZE_OF_CPU_SET_T;
        long[] bits = affinity.toLongArray();
        for (int i = 0; i < bits.length; i++) {
            cpu_mask.__bits[i].setValue(bits[i]);
        }
        return cStdLib.sched_setaffinity(0, size, cpu_mask);
    }

    public void setCpuAffinity(int cpuId) {
        BitSet affinity = new BitSet(Runtime.getRuntime().availableProcessors());
        affinity.set(cpuId);
        if (setCpuAffinity(affinity) != 0) {
            throw new RuntimeException("Could not set CPU affinity for CPU ID " + cpuId);
        }
    }

    public int getCpuId() {
        return cStdLib.sched_getcpu();
    }

    public int getTid() {
        return cStdLib.gettid();
    }

    public void setTid() {
        try {
            int nativeThreadId = getTid();
            final Field tid = Thread.class.getDeclaredField("tid");
            tid.setAccessible(true);
            final Thread thread = Thread.currentThread();
            tid.setLong(thread, nativeThreadId);
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
    }

    public static CpuAffinity getInstance() {
        return instance;
    }
}
