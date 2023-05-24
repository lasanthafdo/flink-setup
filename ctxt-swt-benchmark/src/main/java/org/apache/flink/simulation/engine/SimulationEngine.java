package org.apache.flink.simulation.engine;

import org.apache.flink.simulation.Operator;
import org.apache.flink.simulation.Subtask;

import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class SimulationEngine {
    private boolean running;
    private final ExecutorService executorService;
    private final Set<Operator> currentOperators = new HashSet<>(10);
    private final Set<Subtask> currentSubtasks = new HashSet<>(50);

    public SimulationEngine(int numWorkerThreads) {
        this.executorService = Executors.newFixedThreadPool(numWorkerThreads);
    }

    public void configure() {

    }

    public void start() {
        for (Subtask subtask : currentSubtasks) {
            executorService.submit(subtask);
        }
    }

    public void addOperator(Operator operator) {
        if (!currentOperators.add(operator)) {
            throw new IllegalArgumentException("Operator has already been added!");
        }
        if (!currentSubtasks.addAll(operator.getSubtasks())) {
            throw new IllegalStateException("Different operators have same task(s)!");
        }
    }
}
