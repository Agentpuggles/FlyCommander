package fly.agent;

import java.util.*;
import java.util.concurrent.*;

/** Standalone real Java concurrency checks; no Forge mocks or dependencies. */
public final class DecisionBrokerTest {
    static void check(boolean ok) { if (!ok) throw new AssertionError(); }
    static DecisionBroker.Request await(DecisionBroker b) throws Exception {
        long end = System.nanoTime() + TimeUnit.SECONDS.toNanos(2);
        while (b.snapshot() == null && System.nanoTime() < end) Thread.sleep(1);
        if (b.snapshot() == null) throw new AssertionError("request was not published");
        return b.snapshot();
    }
    public static void main(String[] args) throws Exception {
        DecisionBroker b = new DecisionBroker();
        var options = List.of(new DecisionBroker.Option("a", "A"), new DecisionBroker.Option("b", "B"));
        ExecutorService worker = Executors.newSingleThreadExecutor();
        try {
            Future<List<String>> answer = worker.submit(() -> b.ask("choice", "Pick", options, 1, 2, 0));
            var request = await(b);
            check(!answer.isDone()); // game thread really blocks
            check(request.id().equals(b.snapshot().id())); // reconnect does not replace request
            check(!b.submit("stale", List.of("a")));
            check(!b.submit(request.id(), List.of("a", "a")));
            check(!b.submit(request.id(), List.of("hidden")));
            check(b.submit(request.id(), List.of("b", "a")));
            check(!b.submit(request.id(), List.of("a")));
            check(answer.get(2, TimeUnit.SECONDS).equals(List.of("b", "a"))); // order preserved
            check(b.snapshot() == null);
            Future<List<String>> cancelled = worker.submit(() -> b.ask("choice", "Pick", options, 1, 1, 0));
            var next = await(b);
            check(!request.id().equals(next.id()));
            check(b.cancel(next.id(), "Game ended"));
            try { cancelled.get(2, TimeUnit.SECONDS); throw new AssertionError(); }
            catch (ExecutionException e) { check(e.getCause() instanceof CancellationException); }
            try { b.ask("choice", "Pick", options, 1, 1, 10); throw new AssertionError(); }
            catch (TimeoutException expected) { check(b.snapshot() == null); }
            Thread.currentThread().interrupt();
            try { b.ask("choice", "Pick", options, 1, 1, 0); throw new AssertionError(); }
            catch (InterruptedException expected) { check(b.snapshot() == null); }
        } finally { worker.shutdownNow(); }
        System.out.println("DecisionBroker lifecycle tests passed");
    }
}
