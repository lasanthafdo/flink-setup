package org.apache.flink.simulation;

public abstract class AbstractSourceOperator extends AbstractOperator {

    public AbstractSourceOperator(String operatorName, int parallelism) {
        super(operatorName, parallelism, true, false);
    }

    @Override
    public void processSingleEvent() throws InterruptedException {
        processEvent(null, outputQueue);
    }
}
