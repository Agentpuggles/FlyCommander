package fly.agent;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.Executors;

/**
 * Minimal HTTP agent exposing Forge's internal state to the Python fly brain.
 *
 * Endpoints:
 *   GET  /health        agent + process status (+ table queue counters)
 *   GET  /observation   structured game-state observation from the fly's seat
 *   GET  /result        results of all completed games so far
 *   GET  /table/queue   physical-table action queue counters
 *   POST /table/events  physical-table action batch (see forge_table_bridge.py)
 *   POST /table/state   full battlefield resync for the table seat
 *
 * The /table/* endpoints are how the camera side of FlyCommander talks to
 * Forge: the table observes, Forge judges. They are only useful when a match
 * was started with a physical seat (system property {@code fly.agent.table}).
 */
public final class AgentServer {
    private static volatile int port = 8791;
    private static volatile boolean ready = false;
    private static volatile String lastResultJson = "{}";
    private static volatile int gamesCompleted = 0;

    private static volatile String lastObservationJson = "{}";
    private static volatile int lastDecision = -1;
    private static volatile String lastAction = "none";
    private static volatile int currentGame = 0;
    private static volatile int decisionsThisGame = 0;

    private static volatile String flyDeckName = "";
    private static volatile java.util.List<String> aiDeckNames = java.util.List.of();

    private static HttpServer server;

    private AgentServer() {}

    public static synchronized void start(int listenPort) throws IOException {
        port = listenPort;
        server = HttpServer.create(new InetSocketAddress("127.0.0.1", port), 0);
        server.createContext("/health", AgentServer::handleHealth);
        server.createContext("/observation", AgentServer::handleObservation);
        server.createContext("/result", AgentServer::handleResult);
        server.createContext("/human/state", ex -> respond(ex, 200, WebHumanSession.active() ? WebHumanSession.snapshotJson() : HumanDecisionChannel.snapshot()));
        server.createContext("/human/decision", AgentServer::handleHumanDecision);
        server.createContext("/table/events", AgentServer::handleTableEvents);
        server.createContext("/table/state", AgentServer::handleTableState);
        server.createContext("/table/queue", AgentServer::handleTableQueue);
        server.setExecutor(Executors.newFixedThreadPool(2));
        server.start();
        System.out.println("[FlyAgent] HTTP agent listening on 127.0.0.1:" + port);
    }

    // ------------------------------------------------------------------
    public static void markReady() { ready = true; }

    public static void setDecks(String flyName, java.util.List<String> aiNames) {
        flyDeckName = flyName;
        aiDeckNames = java.util.List.copyOf(aiNames);
    }

    public static String resolvedHumanDeckName() { return flyDeckName; }
    public static java.util.List<String> resolvedAiDeckNames() { return aiDeckNames; }

    public static void beginGame(int n) {
        currentGame = n;
        decisionsThisGame = 0;
    }

    public static void recordResult(String json) {
        lastResultJson = json;
        gamesCompleted++;
    }

    public static void pushObservation(String json) {
        lastObservationJson = json;
    }

    public static String lastObservationJson() {
        return lastObservationJson;
    }

    public static void recordDecision(int action) {
        lastDecision = action;
        decisionsThisGame++;
    }

    public static int lastDecision() {
        return lastDecision;
    }

    public static void recordAction(String what) {
        lastAction = what;
    }

    // ------------------------------------------------------------------
    private static void handleHealth(HttpExchange ex) throws IOException {
        StringBuilder ai = new StringBuilder();
        for (String n : aiDeckNames) {
            if (ai.length() > 0) ai.append(',');
            ai.append('"').append(n.replace("\\", "\\\\").replace("\"", "'"))
              .append('"');
        }
        String body = "{\"status\":\"" + (ready ? "ready" : "booting")
                + "\",\"port\":" + port
                + ",\"gamesCompleted\":" + gamesCompleted
                + ",\"currentGame\":" + currentGame
                + ",\"decisionsThisGame\":" + decisionsThisGame
                + ",\"lastDecision\":" + lastDecision
                + ",\"lastAction\":\"" + lastAction + "\""
                + ",\"flyDeck\":\"" + flyDeckName + "\""
                + ",\"aiDecks\":[" + ai + "]}";
        respond(ex, 200, body);
    }

    private static void handleObservation(HttpExchange ex) throws IOException {
        if (!ready) {
            respond(ex, 503, "{\"error\":\"agent booting\"}");
            return;
        }
        // Never walk live Forge objects from an HTTP worker.
        String json = lastObservationJson;
        respond(ex, 200, json);
    }

    private static void handleResult(HttpExchange ex) throws IOException {
        respond(ex, 200, lastResultJson);
    }

    // ------------------------------------------------------------------
    // physical table → Forge
    // ------------------------------------------------------------------
    private static void handleTableEvents(HttpExchange ex) throws IOException {
        respond(ex, 410, "{\"error\":\"Scanner mutations disabled. Use human decision prompts; scans do not play cards.\"}");
    }
    private static void handleTableState(HttpExchange ex) throws IOException {
        handleTableEvents(ex);
    }
    private static void handleHumanDecision(HttpExchange ex) throws IOException {
        if (!"POST".equals(ex.getRequestMethod())) {
            respond(ex, 405, "{\"error\":\"POST required\"}"); return;
        }
        try {
            var body = MiniJson.parseObject(readBody(ex));
            boolean ok;
            if (WebHumanSession.active()) {
                ok = WebHumanSession.submit(body);
            } else if (body.get("selected") instanceof java.util.List<?> raw) {
                java.util.List<String> selected = new java.util.ArrayList<>();
                for (Object value : raw) {
                    if (!(value instanceof String)) { selected.clear(); break; }
                    selected.add((String) value);
                }
                ok = !selected.isEmpty() || raw.isEmpty()
                        ? HumanDecisionChannel.submit(String.valueOf(body.get("id")), selected)
                        : false;
            } else {
                ok = body.get("id") instanceof String id
                        && body.get("choice") instanceof String choice
                        && HumanDecisionChannel.submit(id, choice);
            }
            respond(ex, ok ? 200 : 409, ok ? "{\"status\":\"accepted\"}" : "{\"error\":\"Stale, duplicate or invalid decision; refresh state\"}");
        } catch (RuntimeException e) { respond(ex, 400, "{\"error\":\"Invalid decision\"}"); }
    }

    private static void handleTableQueue(HttpExchange ex) throws IOException {
        respond(ex, 200, TableActionQueue.statsJson());
    }

    private static String readBody(HttpExchange ex) throws IOException {
        byte[] raw = ex.getRequestBody().readNBytes(65537);
        if (raw.length > 65536) throw new IllegalArgumentException("body too large");
        return new String(raw, StandardCharsets.UTF_8);
    }

    private static void respond(HttpExchange ex, int code, String body) throws IOException {
        byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
        ex.getResponseHeaders().add("Content-Type", "application/json");
        ex.sendResponseHeaders(code, bytes.length);
        try (OutputStream os = ex.getResponseBody()) {
            os.write(bytes);
        }
    }

    private static String sanitize(String s) {
        return s.replace("\\", "\\\\").replace("\"", "'").replace("\n", " ");
    }
}
