package fly.agent;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Dependency-free JSON reader for the table-action endpoint.
 *
 * The rest of the patch hand-writes JSON, so it never needs a parser; the
 * physical-table bridge posts real JSON, and pulling in a library just for
 * this would couple the patch to whatever the Forge jar happens to bundle.
 * Supports exactly the subset the contract uses: objects, arrays, strings,
 * numbers, booleans and null.
 */
final class MiniJson {

    private final String src;
    private int pos;

    private MiniJson(String src) {
        this.src = src;
    }

    static Object parse(String text) {
        MiniJson p = new MiniJson(text);
        p.skipWs();
        Object value = p.value();
        p.skipWs();
        if (p.pos != text.length()) throw new IllegalArgumentException("trailing JSON");
        return value;
    }

    @SuppressWarnings("unchecked")
    static Map<String, Object> parseObject(String text) {
        Object value = parse(text);
        if (value instanceof Map) {
            return (Map<String, Object>) value;
        }
        throw new IllegalArgumentException("expected a JSON object");
    }

    static String stringify(Object value) {
        if (value == null) return "null";
        if (value instanceof Boolean || value instanceof Number) return value.toString();
        if (value instanceof Map<?, ?> map) {
            List<String> parts = new ArrayList<>();
            for (var e : map.entrySet()) parts.add(stringify(e.getKey().toString()) + ":" + stringify(e.getValue()));
            return "{" + String.join(",", parts) + "}";
        }
        if (value instanceof Iterable<?> items) {
            List<String> parts = new ArrayList<>();
            for (Object item : items) parts.add(stringify(item));
            return "[" + String.join(",", parts) + "]";
        }
        StringBuilder out = new StringBuilder("\"");
        for (char c : value.toString().toCharArray()) {
            if (c == '"' || c == '\\') out.append('\\').append(c);
            else if (c < 32) out.append(String.format("\\u%04x", (int)c));
            else out.append(c);
        }
        return out.append('"').toString();
    }

    // ------------------------------------------------------------------
    private Object value() {
        if (pos >= src.length()) {
            throw new IllegalArgumentException("unexpected end of JSON");
        }
        char c = src.charAt(pos);
        switch (c) {
            case '{': return object();
            case '[': return array();
            case '"': return string();
            case 't': expect("true"); return Boolean.TRUE;
            case 'f': expect("false"); return Boolean.FALSE;
            case 'n': expect("null"); return null;
            default: return number();
        }
    }

    private Map<String, Object> object() {
        Map<String, Object> map = new LinkedHashMap<>();
        pos++;                       // '{'
        skipWs();
        if (peek() == '}') { pos++; return map; }
        while (true) {
            skipWs();
            String key = string();
            skipWs();
            expect(":");
            skipWs();
            map.put(key, value());
            skipWs();
            char c = next();
            if (c == '}') return map;
            if (c != ',') throw new IllegalArgumentException("bad object at " + pos);
        }
    }

    private List<Object> array() {
        List<Object> list = new ArrayList<>();
        pos++;                       // '['
        skipWs();
        if (peek() == ']') { pos++; return list; }
        while (true) {
            skipWs();
            list.add(value());
            skipWs();
            char c = next();
            if (c == ']') return list;
            if (c != ',') throw new IllegalArgumentException("bad array at " + pos);
        }
    }

    private String string() {
        expect("\"");
        StringBuilder sb = new StringBuilder();
        while (true) {
            char c = next();
            if (c == '"') return sb.toString();
            if (c != '\\') { sb.append(c); continue; }
            char esc = next();
            switch (esc) {
                case '"': sb.append('"'); break;
                case '\\': sb.append('\\'); break;
                case '/': sb.append('/'); break;
                case 'b': sb.append('\b'); break;
                case 'f': sb.append('\f'); break;
                case 'n': sb.append('\n'); break;
                case 'r': sb.append('\r'); break;
                case 't': sb.append('\t'); break;
                case 'u':
                    String hex = src.substring(pos, pos + 4);
                    pos += 4;
                    sb.append((char) Integer.parseInt(hex, 16));
                    break;
                default: throw new IllegalArgumentException("bad escape \\" + esc);
            }
        }
    }

    private Double number() {
        int start = pos;
        while (pos < src.length() && "-+.eE0123456789".indexOf(src.charAt(pos)) >= 0) {
            pos++;
        }
        if (start == pos) {
            throw new IllegalArgumentException("bad number at " + pos);
        }
        return Double.valueOf(src.substring(start, pos));
    }

    // ------------------------------------------------------------------
    private char peek() {
        return pos < src.length() ? src.charAt(pos) : '\0';
    }

    private char next() {
        if (pos >= src.length()) throw new IllegalArgumentException("eof");
        return src.charAt(pos++);
    }

    private void expect(String token) {
        if (!src.startsWith(token, pos)) {
            throw new IllegalArgumentException("expected '" + token + "' at " + pos);
        }
        pos += token.length();
    }

    private void skipWs() {
        while (pos < src.length() && Character.isWhitespace(src.charAt(pos))) {
            pos++;
        }
    }
}
