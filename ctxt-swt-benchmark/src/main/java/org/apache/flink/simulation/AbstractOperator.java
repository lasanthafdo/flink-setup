package org.apache.flink.simulation;

import java.util.HashSet;
import java.util.Objects;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.TimeUnit;

public abstract class AbstractOperator implements Operator {
    private boolean enabled = false;
    private final boolean source;
    private final boolean sink;
    private final UUID operatorID = UUID.randomUUID();
    protected final String operatorName;
    private final Integer parallelism;
    protected BlockingQueue<Event> inputQueue;
    protected BlockingQueue<Event> outputQueue;
    private final Set<Subtask> subtasks;

    public AbstractOperator(String operatorName, int parallelism, boolean source, boolean sink) {
        this.operatorName = operatorName;
        this.parallelism = parallelism;
        this.subtasks = new HashSet<>(parallelism);
        this.source = source;
        this.sink = sink;
    }

    public AbstractOperator(String operatorName, int parallelism) {
        this(operatorName, parallelism, false, false);
    }

    @Override
    public void init() {
        for (int subtaskId = 0; subtaskId < parallelism; subtaskId++) {
            Subtask subtask = new Subtask(this, subtaskId);
            subtasks.add(subtask);
        }
        enabled = true;
    }

    @Override
    public void stop() {
        enabled = false;
    }

    @Override
    public void processSingleEvent() throws InterruptedException {
        Event currentEvent = inputQueue.poll(1, TimeUnit.SECONDS);
        if (currentEvent != null) {
            processEvent(currentEvent, outputQueue);
        }
    }

    protected abstract void processEvent(Event inputEvent, BlockingQueue<Event> outputQueue) throws
        InterruptedException;

    @Override
    public void setInput(BlockingQueue<Event> inputQueue) {
        this.inputQueue = inputQueue;
    }

    @Override
    public void setOutput(BlockingQueue<Event> outputQueue) {
        this.outputQueue = outputQueue;
    }

    @Override
    public boolean isEnabled() {
        return enabled;
    }

    @Override
    public String getOperatorName() {
        return operatorName;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        AbstractOperator that = (AbstractOperator) o;
        return Objects.equals(operatorID, that.operatorID) &&
            Objects.equals(operatorName, that.operatorName);
    }

    @Override
    public int hashCode() {
        return Objects.hash(operatorID, operatorName);
    }

    @Override
    public Set<Subtask> getSubtasks() {
        return subtasks;
    }

    @Override
    public boolean isSource() {
        return source;
    }

    @Override
    public boolean isSink() {
        return sink;
    }

    @Override
    public boolean isInputQueueEmpty() {
        return inputQueue.isEmpty();
    }
}
