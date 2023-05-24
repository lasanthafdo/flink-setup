package org.apache.flink.simulation;

public class Event {
    private long timestamp;
    private double doubleVal;

    public Event(long timestamp, double val) {
        this.timestamp = timestamp;
        this.doubleVal = val;
    }
}
