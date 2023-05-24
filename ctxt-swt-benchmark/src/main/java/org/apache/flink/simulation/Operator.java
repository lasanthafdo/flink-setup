package org.apache.flink.simulation;

import java.util.Set;
import java.util.concurrent.BlockingQueue;

public interface Operator {
    void setInput(BlockingQueue<Event> inputQueue);

    void setOutput(BlockingQueue<Event> outputQueue);

    boolean isEnabled();

    void processSingleEvent() throws InterruptedException;

    void init();

    void stop();

    String getOperatorName();

    Set<Subtask> getSubtasks();

    boolean isSource();

    boolean isSink();

    boolean isInputQueueEmpty();
}
