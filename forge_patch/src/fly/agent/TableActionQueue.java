package fly.agent;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Queue of physical-table actions waiting to be applied to the Forge game.
 *
 * The Python bridge (see {@code flycommander/forge_table_bridge.py}) posts
 * batches to {@code POST /table/events}; the HTTP thread only *enqueues* them.
 * Application happens on the game thread, at the seat's next priority, so no
 * Forge state is ever mutated from an HTTP worker. Ordering is preserved and
 * action ids are de-duplicated, so a bridge retry can never double-apply.
 */
final class TableActionQueue {

    /** One physical action: its JSON body plus the id Forge acknowledges. */
    static final class Action {
        final String actionId;
        final String kind;
        final Map<String, Object> fields;

        Action(String actionId, String kind, Map<String, Object> fields) {
            this.actionId = actionId;
            this.kind = kind;
            this.fields = fields;
        }

        String str(String key) {
            Object v = fields.get(key);
            return v == null ? null : String.valueOf(v);
        }

        boolean bool(String key, boolean fallback) {
            Object v = fields.get(key);
            if (v instanceof Boolean) return (Boolean) v;
            if (v instanceof Number) return ((Number) v).doubleValue() != 0;
            if (v instanceof String) return Boolean.parseBoolean((String) v);
            return fallback;
        }

        int intValue(String key, int fallback) {
            Object v = fields.get(key);
            if (v instanceof Number) return ((Number) v).intValue();
            if (v instanceof String) {
                try {
                    return Integer.parseInt((String) v);
                } catch (NumberFormatException ignored) {
                    return fallback;
                }
            }
            return fallback;
        }

        @SuppressWarnings("unchecked")
        Map<String, Object> map(String key) {
            Object v = fields.get(key);
            return v instanceof Map ? (Map<String, Object>) v : null;
        }
    }

    private static final Object LOCK = new Object();
    private static final List<Action> QUEUE = new ArrayList<>();
    private static final Set<String> SEEN_IDS = new LinkedHashSet<>();
    private static final int MAX_QUEUE = 512;
    private static final int MAX_SEEN = 4096;

    private static int appliedTotal = 0;
    private static int rejectedTotal = 0;
    private static int droppedTotal = 0;
    private static int duplicates = 0;
    private static final List<String> RECENT_RESULTS = new ArrayList<>();

    private TableActionQueue() {}

    // ------------------------------------------------------------------
    /** Enqueue one action; returns false when it was a known duplicate. */
    static boolean offer(Action action) {
        synchronized (LOCK) {
            if (action.actionId != null && !action.actionId.isEmpty()) {
                if (SEEN_IDS.contains(action.actionId)) {
                    duplicates++;
                    return false;
                }
                SEEN_IDS.add(action.actionId);
                while (SEEN_IDS.size() > MAX_SEEN) {
                    java.util.Iterator<String> it = SEEN_IDS.iterator();
                    it.next();
                    it.remove();
                }
            }
            QUEUE.add(action);
            while (QUEUE.size() > MAX_QUEUE) {
                QUEUE.remove(0);
                droppedTotal++;
            }
            return true;
        }
    }

    static int size() {
        synchronized (LOCK) {
            return QUEUE.size();
        }
    }

    /** Take everything queued (game thread only). */
    static List<Action> drain() {
        synchronized (LOCK) {
            List<Action> out = new ArrayList<>(QUEUE);
            QUEUE.clear();
            return out;
        }
    }

    static void recordResult(String actionId, String kind, String status, String reason) {
        synchronized (LOCK) {
            if ("applied".equals(status)) {
                appliedTotal++;
            } else {
                rejectedTotal++;
            }
            String entry = status + ":" + kind
                    + (reason == null || reason.isEmpty() ? "" : " (" + reason + ")");
            RECENT_RESULTS.add(entry);
            while (RECENT_RESULTS.size() > 32) {
                RECENT_RESULTS.remove(0);
            }
        }
    }

    static String statsJson() {
        synchronized (LOCK) {
            StringBuilder sb = new StringBuilder();
            sb.append("{\"pending\":").append(QUEUE.size())
              .append(",\"applied\":").append(appliedTotal)
              .append(",\"rejected\":").append(rejectedTotal)
              .append(",\"dropped\":").append(droppedTotal)
              .append(",\"duplicates\":").append(duplicates)
              .append(",\"recent\":[");
            for (int i = 0; i < RECENT_RESULTS.size(); i++) {
                if (i > 0) sb.append(',');
                sb.append('"').append(esc(RECENT_RESULTS.get(i))).append('"');
            }
            sb.append("]}");
            return sb.toString();
        }
    }

    /** Parse a bridge batch body: {tableId, seat, events:[...]}. */
    static List<Action> parseBatch(String body) {
        List<Action> actions = new ArrayList<>();
        Map<String, Object> root = MiniJson.parseObject(body);
        Object events = root.get("events");
        if (!(events instanceof List)) {
            return actions;
        }
        for (Object item : (List<?>) events) {
            if (!(item instanceof Map)) continue;
            @SuppressWarnings("unchecked")
            Map<String, Object> map = (Map<String, Object>) item;
            String id = map.get("actionId") == null ? null : String.valueOf(map.get("actionId"));
            String kind = map.get("action") == null ? null : String.valueOf(map.get("action"));
            if (kind == null) continue;
            actions.add(new Action(id, kind, new LinkedHashMap<>(map)));
        }
        return actions;
    }

    /** Parse {@code POST /table/state} into a single {@code sync_state} action. */
    static Action parseState(String body) {
        Map<String, Object> root = MiniJson.parseObject(body);
        String id = root.get("tableId") == null
                ? "sync" : String.valueOf(root.get("tableId")) + ":sync";
        return new Action(id, "sync_state", new LinkedHashMap<>(root));
    }

    private static String esc(String s) {
        return s.replace("\\", "\\\\").replace("\"", "'").replace("\n", " ");
    }
}
