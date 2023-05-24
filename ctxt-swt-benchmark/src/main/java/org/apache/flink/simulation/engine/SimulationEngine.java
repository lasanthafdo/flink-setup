package org.apache.flink.simulation.engine;

import org.apache.flink.simulation.Event;
import org.apache.flink.simulation.Operator;
import org.apache.flink.simulation.Subtask;

import java.util.*;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;

public class SimulationEngine {
    private boolean running;
    private final int queueSize;
    private final ExecutorService executorService;
    private final Map<String, Operator> currentOperators = new HashMap<>(10);
    private List<String> operatorOrder;
    private final Set<Subtask> currentSubtasks = new HashSet<>(50);

    public SimulationEngine(int numWorkerThreads, int queueSize) {
        this.executorService = Executors.newFixedThreadPool(numWorkerThreads);
        this.queueSize = queueSize;
    }

    public void configure(List<String> operatorOrder) {
        System.out.println("Configuring simulation engine...");
        this.operatorOrder = operatorOrder;
        BlockingQueue<Event> inputForNextOp = null;
        for (String operatorName : operatorOrder) {
            Operator currOp = currentOperators.get(operatorName);
            if (!currOp.isSource()) {
                currOp.setInput(inputForNextOp);
            }
            if (!currOp.isSink()) {
                inputForNextOp = new LinkedBlockingQueue<>(queueSize);
                currOp.setOutput(inputForNextOp);
            }
        }
        for (Operator op : currentOperators.values()) {
            op.init();
            if (!currentSubtasks.addAll(op.getSubtasks())) {
                throw new IllegalStateException("Different operators have same task(s)!");
            }
        }
    }

    public void start() {
        running = true;
        for (Subtask subtask : currentSubtasks) {
            executorService.submit(subtask);
        }
    }

    public void shutdown() {
        System.out.println("Shutting down engine...");
        for (String operatorName : operatorOrder) {
            Operator currOp = currentOperators.get(operatorName);
            if (!currOp.isSource()) {
                while (!currOp.isInputQueueEmpty()) {
                    Thread.yield();
                }
            }
            currOp.stop();
        }
        executorService.shutdown();
        running = false;
    }

    public SimulationEngine addOperator(Operator operator) {
        if (currentOperators.put(operator.getOperatorName(), operator) != null) {
            throw new IllegalArgumentException("Operator has already been added!");
        }
        return this;
    }

    public boolean isRunning() {
        return running;
    }
}
