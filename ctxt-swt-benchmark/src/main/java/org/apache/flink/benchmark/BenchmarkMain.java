package org.apache.flink.benchmark;

import org.apache.commons.cli.*;
import org.apache.flink.simulation.DefaultEventGenerator;
import org.apache.flink.simulation.DefaultSink;
import org.apache.flink.simulation.SimpleTransform;
import org.apache.flink.simulation.engine.SimulationEngine;

import java.util.ArrayList;
import java.util.List;

public class BenchmarkMain {
    public static void main(String[] args) {
        Options options = new Options();

        Option numThreads = new Option("t", "numThreads", true, "Number of worker threads");
        Option queueSize = new Option("q", "queueSize", true, "Size of input queues");
        Option parallelism =
            new Option("p", "parallelism", true, "Parallelism level (number of subtasks for each operator");
        Option duration = new Option("d", "duration", true, "Duration (in seconds) the benchmark should run");
        options.addOption(numThreads);
        options.addOption(queueSize);
        options.addOption(parallelism);
        options.addOption(duration);

        // define parser
        CommandLine cmd;
        CommandLineParser parser = new DefaultParser();
        HelpFormatter helper = new HelpFormatter();

        try {
            cmd = parser.parse(options, args);
            SimulationEngine simEngine = new SimulationEngine(Integer.parseInt(cmd.getOptionValue("numThreads", "10")),
                Integer.parseInt(cmd.getOptionValue("queueSize", "10000")));
            int parallelismLevel = Integer.parseInt(cmd.getOptionValue("parallelism", "2"));
            long benchmarkDuration = Long.parseLong(cmd.getOptionValue("duration", "3")) * 1000L;
            simEngine
                .addOperator(new DefaultEventGenerator("source", parallelismLevel))
                .addOperator(new SimpleTransform("transform", parallelismLevel))
                .addOperator(new DefaultSink("sink", parallelismLevel));
            List<String> opOrder = new ArrayList<>();
            opOrder.add("source");
            opOrder.add("transform");
            opOrder.add("sink");
            simEngine.configure(opOrder);
            simEngine.start();
            try {
                Thread.sleep(benchmarkDuration);
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
            simEngine.shutdown();
            if (!simEngine.isRunning()) {
                System.out.println("Simulation engine successfully shutdown.");
            }
            System.out.println("Benchmark application finished.");
        } catch (ParseException e) {
            System.out.println(e.getMessage());
            helper.printHelp("Usage:", options);
            System.exit(0);
        }
    }
}
