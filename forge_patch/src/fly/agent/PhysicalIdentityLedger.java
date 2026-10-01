package fly.agent;

import java.util.*;

/** Identity boundary only, NOT Magic legality or a zone mutator.
 * Call from game thread. Observations cannot commit game movements.
 */
public final class PhysicalIdentityLedger {
    public record Identity(String physicalId, String printing, Integer forgeId, String zone) {}
    public record Operation(String id, String physicalId, String from, String to, long revision) {}
    private final Map<String, Identity> cards = new LinkedHashMap<>();
    private final Map<String, Operation> operations = new HashMap<>();
    private long revision;

    public void observe(String id, String printing) {
        if (id == null || id.isBlank() || cards.containsKey(id))
            throw new IllegalArgumentException("Missing or already observed physical instance");
        cards.put(id, new Identity(id, printing, null, "unassigned"));
    }
    public void bind(String id, int forgeId, String printing, String authoritativeZone) {
        Identity c = require(id);
        if (printing == null || c.printing() == null || !printing.equals(c.printing())
                || c.forgeId() != null || cards.values().stream().anyMatch(x -> Objects.equals(x.forgeId(), forgeId)))
            throw new IllegalArgumentException("Identity unknown, mismatched, or already bound");
        cards.put(id, new Identity(id, printing, forgeId, authoritativeZone));
        revision++;
    }
    public Operation expect(String id, String from, String to) {
        Identity c = require(id);
        if (c.forgeId() == null || !c.zone().equals(from)) throw new IllegalStateException("Unbound or wrong zone");
        Operation op = new Operation(UUID.randomUUID().toString(), id, from, to, revision);
        operations.put(op.id(), op);
        return op;
    }
    /** Only after Forge confirms movement; never invoked by scanner confirmation. */
    public void commit(String operationId, int forgeId, String authoritativeZone) {
        Operation op = operations.get(operationId);
        if (op == null || op.revision() != revision) throw new IllegalStateException("Stale movement acknowledgement");
        Identity c = require(op.physicalId());
        if (!Objects.equals(c.forgeId(), forgeId) || !op.to().equals(authoritativeZone))
            throw new IllegalArgumentException("Forge identity/zone mismatch");
        cards.put(c.physicalId(), new Identity(c.physicalId(), c.printing(), forgeId, authoritativeZone));
        revision++;
        operations.clear();
    }
    public Identity get(String id) { return require(id); }
    private Identity require(String id) {
        Identity c = cards.get(id);
        if (c == null) throw new IllegalArgumentException("Unknown physical instance");
        return c;
    }
}
