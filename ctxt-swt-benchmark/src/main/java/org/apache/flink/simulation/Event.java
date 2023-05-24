package org.apache.flink.simulation;

public class Event {
    private final long timestamp;
    private final double doubleVal;

    public Event(long timestamp, double val) {
        this.timestamp = timestamp;
        this.doubleVal = val;
    }

    public long getTimestamp() {
        return timestamp;
    }

    public double getDoubleVal() {
        return doubleVal;
    }
}
