package org.apache.flink.benchmark;

import org.apache.flink.simulation.DefaultEventGenerator;
import org.apache.flink.simulation.DefaultSink;
import org.apache.flink.simulation.SimpleTransform;
import org.apache.flink.simulation.engine.SimulationEngine;

import java.util.ArrayList;
import java.util.List;

public class BenchmarkMain {
    public static void main(String[] args) {
        SimulationEngine simEngine = new SimulationEngine(10, 10000);
        simEngine
            .addOperator(new DefaultEventGenerator("source", 2))
            .addOperator(new SimpleTransform("transform", 2))
            .addOperator(new DefaultSink("sink", 2));
        List<String> opOrder = new ArrayList<>();
        opOrder.add("source");
        opOrder.add("transform");
        opOrder.add("sink");
        simEngine.configure(opOrder);
        simEngine.start();
        try {
            Thread.sleep(3000);
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
        simEngine.shutdown();
        if (!simEngine.isRunning()) {
            System.out.println("Simulation engine successfully shutdown.");
        }
        System.out.println("Benchmark application finished.");
    }
}
