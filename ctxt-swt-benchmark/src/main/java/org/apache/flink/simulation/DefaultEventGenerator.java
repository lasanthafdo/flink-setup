package org.apache.flink.simulation;

import java.util.Random;
import java.util.concurrent.BlockingQueue;

public class DefaultEventGenerator extends AbstractSourceOperator {

    private final Random rand = new Random(System.currentTimeMillis());

    public DefaultEventGenerator(String operatorName, int parallelism) {
        super(operatorName, parallelism);
    }

    @Override
    protected void processEvent(Event inputEvent, BlockingQueue<Event> outputQueue) {
        new Event(System.currentTimeMillis(), rand.nextLong() / 7.123);
    }

}
