import java.lang.management.ManagementFactory;
import java.lang.management.RuntimeMXBean;
import java.lang.management.ThreadInfo;
import java.lang.management.ThreadMXBean;
import java.util.Arrays;
import java.util.BitSet;

public class MicroBenchmarks {
    public static void main(String[] args) {
        long someLong = 55L;
        long[] someLongArray = new long[]{someLong};
        System.out.println(Arrays.toString(someLongArray));
        BitSet bitSet = BitSet.valueOf(someLongArray);
        printBitset(bitSet);

        RuntimeMXBean runtimeBean = ManagementFactory.getRuntimeMXBean();
        System.out.println("Version: " + runtimeBean.getSpecVersion());
        System.out.println("Process ID: " + ProcessHandle.current().pid());
        Thread myThread = new Thread(new MyRunnable());
        myThread.start();
        ThreadMXBean threadBean = ManagementFactory.getThreadMXBean();
        ThreadInfo threadInfo = threadBean.getThreadInfo(myThread.getId());
        System.out.println(threadInfo.toString());
        ProcessHandle.current().descendants()
            .forEach(processHandle -> System.out.println("Thread PID from top :" + processHandle.pid()));
        try {
            myThread.join();
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
    }

    public static void printBitset(BitSet b) {
        StringBuilder s = new StringBuilder();
        for (int i = 0; i < b.length(); i++) {
            s.append(b.get(i) ? 1 : 0);
        }

        System.out.println(s);
    }

    private static class MyRunnable implements Runnable {

        @Override
        public void run() {
            System.out.println("Thread ID :" + Thread.currentThread().getId());
            System.out.println("Thread TID: " + CpuAffinity.getInstance().getTid());
            CpuAffinity.getInstance().setTid();
            System.out.println("New Thread ID :" + Thread.currentThread().getId());
            try {
                Thread.sleep(1000);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }
}
