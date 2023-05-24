package org.apache.flink.simulation;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.atomic.AtomicLong;

public class DefaultSink extends AbstractSinkOperator {

    private final AtomicLong processedCount = new AtomicLong(0L);

    public DefaultSink(String operatorName, int parallelism) {
        super(operatorName, parallelism);
    }

    @Override
    protected void processEvent(Event inputEvent, BlockingQueue<Event> outputQueue) throws InterruptedException {
        LocalDateTime dateTime = LocalDateTime.now();
        String currentTime = dateTime.format(DateTimeFormatter.ISO_LOCAL_DATE_TIME);
        System.out.println(
            "[" + currentTime + "] Processed event " + processedCount.incrementAndGet() +
                " with value " + inputEvent.getDoubleVal() + " in " +
                (System.currentTimeMillis() - inputEvent.getTimestamp()) + " ms");
    }

}
