package org.apache.flink.simulation;

import java.util.HashSet;
import java.util.Objects;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.BlockingQueue;

public abstract class AbstractOperator implements Operator {
    private boolean enabled = false;
    private final UUID operatorID = UUID.randomUUID();
    protected final String operatorName;
    private final Integer parallelism;
    private BlockingQueue<Event> inputQueue;
    protected BlockingQueue<Event> outputQueue;
    private final Set<Subtask> subtasks;

    public AbstractOperator(String operatorName, int parallelism) {
        this.operatorName = operatorName;
        this.parallelism = parallelism;
        this.subtasks = new HashSet<>(parallelism);
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
    public void processSingleEvent() throws InterruptedException {
        Event currentEvent = inputQueue.take();
        processEvent(currentEvent, outputQueue);
    }

    protected abstract void processEvent(Event inputEvent, BlockingQueue<Event> outputQueue);

    @Override
    public Operator setInput(BlockingQueue<Event> inputQueue) {
        this.inputQueue = inputQueue;
        return this;
    }

    @Override
    public Operator setOutput(BlockingQueue<Event> outputQueue) {
        this.outputQueue = outputQueue;
        return this;
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
}
