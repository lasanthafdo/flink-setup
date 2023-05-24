package org.apache.flink.simulation;

import java.util.concurrent.BlockingQueue;

public class SimpleTransform extends AbstractOperator {
    public SimpleTransform(String operatorName, int parallelism) {
        super(operatorName, parallelism);
    }

    @Override
    protected void processEvent(Event inputEvent, BlockingQueue<Event> outputQueue) throws InterruptedException {
        outputQueue.put(new Event(inputEvent.getTimestamp(), inputEvent.getDoubleVal() * 612.51 / 3.1415926535));
    }
}
