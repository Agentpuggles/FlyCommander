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
 * Endpoints (all GET):
 *   /health      agent + process status
 *   /observation structured game-state observation from the fly's seat
 *   /result      results of all completed games so far
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
        String json;
        try {
            json = GameObserver.snapshotJson(AgentGameState.currentGame(),
                    AgentGameState.flyPlayer());
        } catch (Throwable t) {
            json = "{\"error\":\"snapshot failed: "
                    + sanitize(String.valueOf(t)) + "\"}";
        }
        respond(ex, 200, json);
    }

    private static void handleResult(HttpExchange ex) throws IOException {
        respond(ex, 200, lastResultJson);
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
