package fly.agent;

import java.util.List;
import java.util.Map;
import java.util.UUID;

/** One outstanding decision. HTTP threads exchange JSON only, never Forge objects. */
final class HumanDecisionChannel {
    private static String published = "{\"status\":\"waiting_for_game\"}";
    private static String pendingId;
    private static List<String> allowed = List.of();
    private static String answer;

    static synchronized String snapshot() { return published; }

    static synchronized String ask(Map<String, Object> state, String kind,
                                   String message, List<String> choices) {
        pendingId = UUID.randomUUID().toString();
        allowed = List.copyOf(choices);
        answer = null;
        state.put("status", "awaiting_decision");
        state.put("prompt", Map.of("id", pendingId, "kind", kind,
                "message", message, "choices", allowed));
        published = MiniJson.stringify(state);
        while (answer == null) {
            try { HumanDecisionChannel.class.wait(); }
            catch (InterruptedException e) {
                pendingId = null;
                published = "{\"status\":\"stopped\",\"error\":\"game interrupted\"}";
                Thread.currentThread().interrupt();
                throw new IllegalStateException("Human decision interrupted", e);
            }
        }
        String result = answer;
        pendingId = null;
        state.put("status", "processing");
        state.remove("prompt");
        published = MiniJson.stringify(state);
        return result;
    }

    static synchronized boolean submit(String id, String choice) {
        if (pendingId == null || !pendingId.equals(id) || answer != null
                || !allowed.contains(choice)) return false;
        answer = choice;
        HumanDecisionChannel.class.notifyAll();
        return true;
    }

    static synchronized void finish(Map<String, Object> state) {
        pendingId = null;
        state.put("status", "finished");
        published = MiniJson.stringify(state);
    }
}
