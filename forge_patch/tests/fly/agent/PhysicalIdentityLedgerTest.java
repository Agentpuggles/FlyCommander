package fly.agent;

public final class PhysicalIdentityLedgerTest {
    static void rejects(Runnable r) {
        try { r.run(); } catch (IllegalArgumentException | IllegalStateException expected) { return; }
        throw new AssertionError("Expected rejection");
    }
    public static void main(String[] args) {
        var ledger = new PhysicalIdentityLedger();
        ledger.observe("unknown", null);
        rejects(() -> ledger.bind("unknown", 1, "SET:1", "Hand"));
        ledger.observe("copy1", "SET:1"); ledger.observe("copy2", "SET:1");
        ledger.bind("copy1", 1, "SET:1", "Hand");
        rejects(() -> ledger.bind("copy2", 1, "SET:1", "Hand"));
        ledger.bind("copy2", 2, "SET:1", "Hand");
        var op = ledger.expect("copy1", "Hand", "Battlefield");
        if (!ledger.get("copy1").zone().equals("Hand")) throw new AssertionError();
        rejects(() -> ledger.commit(op.id(), 2, "Battlefield"));
        ledger.commit(op.id(), 1, "Battlefield");
        rejects(() -> ledger.commit(op.id(), 1, "Battlefield"));
        if (!ledger.get("copy2").zone().equals("Hand")) throw new AssertionError();
        System.out.println("Physical identity ledger tests passed");
    }
}
