package org.apache.flink.simulation;

import java.util.Set;
import java.util.concurrent.BlockingQueue;

public interface Operator {
    Operator setInput(BlockingQueue<Event> inputQueue);

    Operator setOutput(BlockingQueue<Event> outputQueue);

    boolean isEnabled();

    void processSingleEvent() throws InterruptedException;

    void init();

    String getOperatorName();

    Set<Subtask> getSubtasks();
}
