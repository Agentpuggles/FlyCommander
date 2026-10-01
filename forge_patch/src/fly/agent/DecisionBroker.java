package fly.agent;

import java.util.*;
import java.util.concurrent.CancellationException;
import java.util.concurrent.TimeoutException;

/** Per-human, single outstanding request. No Forge objects cross this boundary. */
public final class DecisionBroker {
    public record Option(String id, String label) {}
    public record Request(String id, String kind, String message, List<Option> options,
                          int min, int max) {}
    private Request pending;
    private List<String> answer;
    private String cancellation;

    public synchronized Request snapshot() { return pending; }

    public synchronized List<String> ask(String kind, String message, List<Option> options,
                                          int min, int max, long timeoutMillis)
            throws InterruptedException, TimeoutException {
        if (pending != null) throw new IllegalStateException("Decision already pending");
        if (min < 0 || max < min || max > options.size() || timeoutMillis < 0)
            throw new IllegalArgumentException("Invalid decision bounds");
        if (options.stream().map(Option::id).distinct().count() != options.size())
            throw new IllegalArgumentException("Duplicate option IDs");
        pending = new Request(UUID.randomUUID().toString(), kind, message,
                List.copyOf(options), min, max);
        answer = null;
        cancellation = null;
        long start = System.nanoTime();
        try {
            while (answer == null && cancellation == null) {
                if (timeoutMillis == 0) wait();
                else {
                    long left = timeoutMillis - (System.nanoTime() - start) / 1_000_000;
                    if (left <= 0) throw new TimeoutException("Human decision timed out; do not auto-pass");
                    wait(left);
                }
            }
            if (cancellation != null) throw new CancellationException(cancellation);
            return answer;
        } finally {
            pending = null;
            answer = null;
        }
    }

    /** Reconnects read the same request ID. Duplicate/stale submissions fail. */
    public synchronized boolean submit(String id, List<String> selected) {
        if (pending == null || !pending.id().equals(id) || answer != null || cancellation != null
                || selected == null || selected.size() < pending.min() || selected.size() > pending.max()
                || new HashSet<>(selected).size() != selected.size()) return false;
        Set<String> allowed = new HashSet<>();
        for (Option o : pending.options()) allowed.add(o.id());
        if (!allowed.containsAll(selected)) return false;
        answer = List.copyOf(selected);
        notifyAll();
        return true;
    }

    public synchronized boolean cancel(String id, String reason) {
        if (pending == null || !pending.id().equals(id) || answer != null || cancellation != null) return false;
        cancellation = reason == null ? "Game cancelled" : reason;
        notifyAll();
        return true;
    }
}
