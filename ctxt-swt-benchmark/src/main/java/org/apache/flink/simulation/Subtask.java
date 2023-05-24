package org.apache.flink.simulation;

import java.util.Objects;

public class Subtask implements Runnable {
    private final Operator owner;
    private final Integer subtaskID;

    public Subtask(Operator owner, int subtaskID) {
        this.owner = owner;
        this.subtaskID = subtaskID;
    }

    @Override
    public void run() {
        try {
            while (owner.isEnabled()) {
                try {
                    owner.processSingleEvent();
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Subtask subtask = (Subtask) o;
        return Objects.equals(owner, subtask.owner) && Objects.equals(subtaskID, subtask.subtaskID);
    }

    @Override
    public int hashCode() {
        return Objects.hash(owner, subtaskID);
    }

    @Override
    public String toString() {
        return owner.getOperatorName() + "-" + subtaskID;
    }
}
