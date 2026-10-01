package fly.agent;

import java.util.*;

/** HTTP-safe publication point. Never traverses Forge game objects. */
public final class WebHumanSession {
    private static volatile DecisionBroker broker;
    private static volatile String fault;
    public static void register(DecisionBroker value) { broker = value; fault = null; }
    public static boolean active() { return broker != null; }
    public static void fault(String message) {
        fault = message;
        DecisionBroker b = broker;
        if (b != null && b.snapshot() != null) b.cancel(b.snapshot().id(), message);
    }
    public static String snapshotJson() {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("mode", Boolean.getBoolean("fly.agent.controllerTest") ? "controller-test-digital-hand" : "physical-development");
        if (fault != null) { out.put("status", "blocked"); out.put("error", fault); }
        else {
            DecisionBroker b = broker;
            DecisionBroker.Request r = b == null ? null : b.snapshot();
            out.put("status", r == null ? "waiting_for_forge" : "awaiting_decision");
            if (r != null) {
                List<Object> options = new ArrayList<>();
                for (var o : r.options()) options.add(Map.of("id", o.id(), "label", o.label()));
                out.put("prompt", Map.of("id", r.id(), "kind", r.kind(), "message", r.message(),
                        "options", options, "min", r.min(), "max", r.max()));
            }
        }
        return MiniJson.stringify(out);
    }
    public static boolean submit(Map<String, Object> body) {
        DecisionBroker b = broker;
        if (b == null || fault != null || !(body.get("id") instanceof String id)
                || !(body.get("selected") instanceof List<?> values)) return false;
        List<String> ids = new ArrayList<>();
        for (Object v : values) { if (!(v instanceof String)) return false; ids.add((String)v); }
        return b.submit(id, ids);
    }
}
