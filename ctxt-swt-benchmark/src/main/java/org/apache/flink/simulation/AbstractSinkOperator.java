package org.apache.flink.simulation;

import java.util.concurrent.TimeUnit;

public abstract class AbstractSinkOperator extends AbstractOperator {

    public AbstractSinkOperator(String operatorName, int parallelism) {
        super(operatorName, parallelism, false, true);
    }

    @Override
    public void processSingleEvent() throws InterruptedException {
        Event currentEvent = inputQueue.poll(1, TimeUnit.SECONDS);
        if (currentEvent != null) {
            processEvent(currentEvent, null);
        }
    }
}
