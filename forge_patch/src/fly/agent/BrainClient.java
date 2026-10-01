package fly.agent;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;

/**
 * Bridge to the Python fly brain. Posts the observation to the brain's
 * /decide endpoint and expects {"action": <index>} back. On any failure
 * returns -1 so the seat can fall back to stock Forge AI.
 */
public final class BrainClient {
    private static final HttpClient HTTP = HttpClient.newBuilder()
            .connectTimeout(Duration.ofMillis(500))
            .build();
    private static final Duration TIMEOUT = Duration.ofSeconds(3);
    private static volatile String brainUrl = "http://127.0.0.1:8792/decide";

    private BrainClient() {}

    public static void setBrainUrl(String url) {
        if (url != null && !url.isBlank()) {
            brainUrl = url;
        }
    }

    /** Returns macro-action index 0..3, or -1 if the brain is unreachable. */
    public static int queryAction(String observationJson, String context) {
        try {
            String body = "{\"observation\":" + (observationJson == null ? "{}" : observationJson)
                    + ",\"context\":\"" + (context == null ? "" : context) + "\"}";
            HttpRequest req = HttpRequest.newBuilder()
                    .uri(URI.create(brainUrl))
                    .timeout(TIMEOUT)
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(body))
                    .build();
            HttpResponse<String> resp = HTTP.send(req, HttpResponse.BodyHandlers.ofString());
            if (resp.statusCode() != 200) {
                return -1;
            }
            return parseAction(resp.body());
        } catch (Exception e) {
            return -1;
        }
    }

    private static int parseAction(String json) {
        if (json == null) return -1;
        java.util.regex.Matcher m = java.util.regex.Pattern
                .compile("\"action\"\\s*:\\s*(-?\\d+)").matcher(json);
        return m.find() ? Integer.parseInt(m.group(1)) : -1;
    }
}
