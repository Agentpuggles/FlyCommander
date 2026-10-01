package fly.agent;

import java.util.*;
import java.util.UUID;

/** One outstanding browser decision. HTTP threads exchange JSON only, never Forge objects. */
final class HumanDecisionChannel {
    private static String published = "{\"status\":\"waiting_for_game\"}";
    private static String pendingId;
    private static List<String> allowed = List.of();
    private static int minSelections;
    private static int maxSelections;
    private static List<String> answer;
    private static String fault;

    private HumanDecisionChannel() { }

    static synchronized String snapshot() { return published; }

    /** Compatibility helper for a prompt with plain string choices. */
    static synchronized String ask(Map<String, Object> state, String kind,
                                   String message, List<String> choices) {
        if (choices == null || choices.isEmpty()) {
            throw new IllegalArgumentException("A decision needs at least one choice");
        }
        begin(state, kind, message);
        allowed = List.copyOf(choices);
        minSelections = 1;
        maxSelections = 1;
        Map<String, Object> prompt = prompt(kind, message);
        prompt.put("choices", allowed);
        state.put("prompt", prompt);
        published = MiniJson.stringify(state);
        List<String> selected = awaitAnswer();
        return selected.get(0);
    }

    /** Ask for an ordered selection from a set of legal, opaque-ID options. */
    static synchronized List<String> askOptions(Map<String, Object> state, String kind,
                                                 String message, List<DecisionBroker.Option> options,
                                                 int min, int max) {
        if (options == null || min < 0 || max < min || max > options.size()) {
            throw new IllegalArgumentException("Invalid decision bounds");
        }
        List<String> ids = new ArrayList<>();
        List<Object> publicOptions = new ArrayList<>();
        for (DecisionBroker.Option option : options) {
            if (option == null || option.id() == null || option.id().isEmpty()
                    || option.label() == null || !ids.add(option.id())) {
                throw new IllegalArgumentException("Invalid or duplicate decision option");
            }
            publicOptions.add(Map.of("id", option.id(), "label", option.label()));
        }
        begin(state, kind, message);
        allowed = List.copyOf(ids);
        minSelections = min;
        maxSelections = max;
        Map<String, Object> prompt = prompt(kind, message);
        prompt.put("options", publicOptions);
        prompt.put("min", min);
        prompt.put("max", max);
        state.put("prompt", prompt);
        published = MiniJson.stringify(state);
        return awaitAnswer();
    }

    private static void begin(Map<String, Object> state, String kind, String message) {
        if (pendingId != null) throw new IllegalStateException("A human decision is already pending");
        pendingId = UUID.randomUUID().toString();
        answer = null;
        fault = null;
        state.put("status", "awaiting_decision");
        state.remove("prompt");
    }

    private static Map<String, Object> prompt(String kind, String message) {
        Map<String, Object> prompt = new LinkedHashMap<>();
        prompt.put("id", pendingId);
        prompt.put("kind", kind);
        prompt.put("message", message);
        return prompt;
    }

    private static List<String> awaitAnswer() {
        while (answer == null && fault == null) {
            try {
                HumanDecisionChannel.class.wait();
            } catch (InterruptedException e) {
                pendingId = null;
                allowed = List.of();
                published = "{\"status\":\"stopped\",\"error\":\"game interrupted\"}";
                Thread.currentThread().interrupt();
                throw new IllegalStateException("Human decision interrupted", e);
            }
        }
        if (fault != null) {
            String message = fault;
            pendingId = null;
            allowed = List.of();
            answer = null;
            throw new IllegalStateException(message);
        }
        List<String> result = answer;
        answer = null;
        pendingId = null;
        allowed = List.of();
        stateAfterDecision();
        return result;
    }

    private static void stateAfterDecision() {
        try {
            Map<String, Object> state = MiniJson.parseObject(published);
            state.remove("prompt");
            state.put("status", "processing");
            published = MiniJson.stringify(state);
        } catch (RuntimeException ignored) {
            published = "{\"status\":\"processing\"}";
        }
    }

    /** Publish an immutable public table snapshot between prompts. */
    static synchronized void publish(Map<String, Object> state) {
        if (pendingId != null || state == null) return;
        state.put("status", "running");
        state.remove("prompt");
        published = MiniJson.stringify(state);
    }

    static synchronized boolean submit(String id, String choice) {
        if (fault != null || pendingId == null || !pendingId.equals(id) || answer != null
                || minSelections != 1 || maxSelections != 1
                || !allowed.contains(choice)) return false;
        answer = List.of(choice);
        HumanDecisionChannel.class.notifyAll();
        return true;
    }

    static synchronized boolean submit(String id, List<String> selections) {
        if (fault != null || pendingId == null || !pendingId.equals(id) || answer != null
                || selections == null || selections.size() < minSelections
                || selections.size() > maxSelections
                || new HashSet<>(selections).size() != selections.size()
                || !allowed.containsAll(selections)) return false;
        answer = List.copyOf(selections);
        HumanDecisionChannel.class.notifyAll();
        return true;
    }

    static synchronized void fault(String message) {
        fault = message == null ? "Forge game stopped unexpectedly" : message;
        pendingId = null;
        allowed = List.of();
        answer = null;
        try {
            Map<String, Object> state = MiniJson.parseObject(published);
            state.remove("prompt");
            state.put("status", "error");
            state.put("error", fault);
            published = MiniJson.stringify(state);
        } catch (RuntimeException ignored) {
            published = MiniJson.stringify(Map.of("status", "error", "error", fault));
        }
        HumanDecisionChannel.class.notifyAll();
    }

    static synchronized void finish(Map<String, Object> state) {
        pendingId = null;
        allowed = List.of();
        answer = null;
        fault = null;
        if (state == null) state = new LinkedHashMap<>();
        state.put("status", "finished");
        state.remove("prompt");
        published = MiniJson.stringify(state);
        HumanDecisionChannel.class.notifyAll();
    }
}
