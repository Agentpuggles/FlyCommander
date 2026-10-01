package fly.agent;

import java.util.*;
import java.util.concurrent.CancellationException;
import java.util.concurrent.TimeoutException;

/** Per-human, single outstanding request. No Forge objects cross this boundary. */
public final class DecisionBroker {
    public record Option(String id, String label) {}
    public record Request(String id, String kind, String message, List<Option> options,
                          int min, int max, List<Option> actions,
                          String inputType, Integer inputMin, Integer inputMax) {}
    public record Answer(List<String> selected, String action) {}

    private Request pending;
    private Answer answer;
    private String cancellation;

    public synchronized Request snapshot() { return pending; }

    public synchronized List<String> ask(String kind, String message, List<Option> options,
                                          int min, int max, long timeoutMillis)
            throws InterruptedException, TimeoutException {
        return askDecision(kind, message, options, min, max, List.of(), null, null, null,
                timeoutMillis).selected();
    }

    /** A multiple-choice prompt with explicit confirm/cancel controls. */
    public synchronized Answer askWithActions(String kind, String message, List<Option> options,
                                               int min, int max, List<Option> actions,
                                               long timeoutMillis)
            throws InterruptedException, TimeoutException {
        return askDecision(kind, message, options, min, max, actions, null, null, null,
                timeoutMillis);
    }

    /** A Forge numeric choice. The client submits one decimal integer, still as an opaque selection ID. */
    public synchronized int askInteger(String kind, String message, int min, int max,
                                        long timeoutMillis)
            throws InterruptedException, TimeoutException {
        Answer response = askDecision(kind, message, List.of(), 1, 1, List.of(),
                "integer", min, max, timeoutMillis);
        try {
            return Integer.parseInt(response.selected().get(0));
        } catch (RuntimeException invalid) {
            throw new IllegalStateException("Invalid numeric response returned by the decision broker", invalid);
        }
    }

    /** A Forge text/name choice. The client submits a single bounded string. */
    public synchronized String askText(String kind, String message, int minLength, int maxLength,
                                       long timeoutMillis)
            throws InterruptedException, TimeoutException {
        Answer response = askDecision(kind, message, List.of(), 1, 1, List.of(),
                "text", minLength, maxLength, timeoutMillis);
        return response.selected().get(0);
    }

    private synchronized Answer askDecision(String kind, String message, List<Option> options,
                                             int min, int max, List<Option> actions,
                                             String inputType, Integer inputMin, Integer inputMax,
                                             long timeoutMillis)
            throws InterruptedException, TimeoutException {
        if (pending != null) throw new IllegalStateException("Decision already pending");
        if (options == null || actions == null || timeoutMillis < 0) {
            throw new IllegalArgumentException("Invalid decision bounds");
        }
        if (inputType == null) {
            if (min < 0 || max < min || max > options.size()) {
                throw new IllegalArgumentException("Invalid decision bounds");
            }
        } else if (!options.isEmpty() || !actions.isEmpty() || min != 1 || max != 1
                || inputMin == null || inputMax == null || inputMin > inputMax) {
            throw new IllegalArgumentException("Invalid typed input bounds");
        }
        if (options.stream().map(Option::id).distinct().count() != options.size()
                || actions.stream().map(Option::id).distinct().count() != actions.size()) {
            throw new IllegalArgumentException("Duplicate option IDs");
        }
        if (options.stream().anyMatch(option -> actions.stream().anyMatch(action -> action.id().equals(option.id())))) {
            throw new IllegalArgumentException("Choice and action IDs must be distinct");
        }
        pending = new Request(UUID.randomUUID().toString(), kind, message,
                List.copyOf(options), min, max, List.copyOf(actions), inputType, inputMin, inputMax);
        answer = null;
        cancellation = null;
        System.out.println("[HumanBroker] pending id=" + pending.id() + " kind=" + kind);
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
        return submit(id, selected, null);
    }

    public synchronized boolean submit(String id, List<String> selected, String action) {
        if (pending == null || !pending.id().equals(id) || answer != null || cancellation != null
                || selected == null || new HashSet<>(selected).size() != selected.size()) return false;

        if (pending.inputType() != null) {
            if (action != null || selected.size() != 1) return false;
            String value = selected.get(0);
            if ("integer".equals(pending.inputType())) {
                try {
                    int number = Integer.parseInt(value);
                    if (number < pending.inputMin() || number > pending.inputMax()) return false;
                } catch (RuntimeException invalid) {
                    return false;
                }
            } else if ("text".equals(pending.inputType())) {
                if (value.length() < pending.inputMin() || value.length() > pending.inputMax()) return false;
            } else {
                return false;
            }
        } else if (action != null) {
            Option submittedAction = pending.actions().stream()
                    .filter(option -> option.id().equals(action)).findFirst().orElse(null);
            if (submittedAction == null) return false;
            if ("cancel".equals(action) && !selected.isEmpty()) return false;
            if ("confirm".equals(action)
                    && (selected.size() < pending.min() || selected.size() > pending.max())) return false;
            if (!allowedChoiceIds().containsAll(selected)) return false;
        } else {
            if (selected.size() < pending.min() || selected.size() > pending.max()
                    || !allowedChoiceIds().containsAll(selected)) return false;
        }

        answer = new Answer(List.copyOf(selected), action);
        System.out.println("[HumanBroker] accepted id=" + id + (action == null ? "" : " action=" + action));
        notifyAll();
        return true;
    }

    private Set<String> allowedChoiceIds() {
        Set<String> allowed = new HashSet<>();
        for (Option option : pending.options()) allowed.add(option.id());
        return allowed;
    }

    public synchronized boolean cancel(String id, String reason) {
        if (pending == null || !pending.id().equals(id) || answer != null || cancellation != null) return false;
        cancellation = reason == null ? "Game cancelled" : reason;
        notifyAll();
        return true;
    }
}
