package fly.agent;

import forge.game.Game;
import forge.game.player.Player;
import java.util.*;

/** HTTP-safe publication point. Live Forge objects stay on the game thread. */
public final class WebHumanSession {
    private static volatile DecisionBroker broker;
    private static volatile String fault;
    private static volatile boolean finished;
    private static volatile String statusMessage = "";
    private static volatile Map<String, Object> gameState = Map.of();

    private WebHumanSession() { }

    public static void register(DecisionBroker value) {
        broker = value;
        fault = null;
        finished = false;
        statusMessage = "";
        gameState = Map.of();
    }

    public static void updateMessage(String message) {
        statusMessage = message == null ? "" : message;
    }

    /** Build the immutable browser snapshot on the Forge game thread, never in an HTTP handler. */
    public static void updateGame(Game game, Player human) {
        if (game == null || human == null) return;
        gameState = Map.copyOf(TableSnapshot.capture(game, human));
    }

    public static void finish(Game game, Player human, boolean humanWon, String reason) {
        updateGame(game, human);
        Map<String, Object> terminal = new LinkedHashMap<>(gameState);
        terminal.put("gameOver", true);
        terminal.put("humanWon", humanWon);
        terminal.put("result", reason == null ? "" : reason);
        gameState = Map.copyOf(terminal);
        finished = true;
    }

    public static boolean active() { return broker != null; }

    public static void fault(String message) {
        fault = message;
        DecisionBroker b = broker;
        if (b != null && b.snapshot() != null) b.cancel(b.snapshot().id(), message);
    }

    public static String snapshotJson() {
        Map<String, Object> out = new LinkedHashMap<>(gameState);
        out.put("mode", "forge-digital-human");
        out.put("physicalSync", "not-installed");
        if (!statusMessage.isBlank()) out.put("message", statusMessage);
        if (fault != null) {
            out.put("status", "blocked");
            out.put("error", fault);
        } else {
            DecisionBroker b = broker;
            DecisionBroker.Request r = b == null ? null : b.snapshot();
            out.put("status", finished ? "finished" : r == null ? "waiting_for_forge" : "awaiting_decision");
            if (r != null) {
                List<Object> options = optionsJson(r.options());
                List<Object> actions = optionsJson(r.actions());
                Map<String, Object> prompt = new LinkedHashMap<>();
                prompt.put("id", r.id());
                prompt.put("kind", r.kind());
                prompt.put("message", r.message());
                prompt.put("options", options);
                prompt.put("min", r.min());
                prompt.put("max", r.max());
                if (!actions.isEmpty()) prompt.put("actions", actions);
                if (r.inputType() != null) {
                    prompt.put("input", Map.of("type", r.inputType(), "min", r.inputMin(), "max", r.inputMax()));
                }
                out.put("prompt", prompt);
            } else {
                out.remove("prompt");
            }
        }
        return MiniJson.stringify(out);
    }

    private static List<Object> optionsJson(List<DecisionBroker.Option> source) {
        List<Object> options = new ArrayList<>();
        for (DecisionBroker.Option option : source) {
            options.add(Map.of("id", option.id(), "label", option.label()));
        }
        return options;
    }

    public static boolean submit(Map<String, Object> body) {
        DecisionBroker b = broker;
        if (b == null || fault != null || !(body.get("id") instanceof String id)
                || !(body.get("selected") instanceof List<?> values)) return false;
        List<String> ids = new ArrayList<>();
        for (Object value : values) {
            if (!(value instanceof String)) return false;
            ids.add((String) value);
        }
        String action = body.get("action") instanceof String text ? text : null;
        return b.submit(id, ids, action);
    }
}
