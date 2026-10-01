package fly.agent;

import forge.deck.Deck;
import forge.deck.DeckgenUtil;
import forge.deck.DeckProxy;
import forge.deck.io.DeckSerializer;
import forge.model.CardCollections;
import forge.model.FModel;
import forge.util.MyRandom;
import forge.util.storage.IStorage;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Deck resolution via Forge's own deck storage — the same pool the normal
 * "Commander → Commander Decks" UI offers and randomizes AI decks from.
 *
 * A deck spec may be:
 *   "random"          Forge picks uniformly from its Commander deck pool
 *                     (DeckProxy.getAllCommanderDecks / MyRandom).
 *   "<deck name>"     looked up in Forge's Commander deck storage
 *                     (FModel.getDecks().getCommander()).
 *   "<path>.dck"      explicit deck file (back-compat; tried as a file first,
 *                     then as a pool name with the extension stripped).
 *
 * This class deliberately contains no filesystem scanning: everything goes
 * through Forge's CardCollections/DeckProxy machinery.
 */
public final class DeckResolver {

    private DeckResolver() {}

    /** Resolve one spec; exits on failure with a clear message. */
    public static Deck resolve(String spec) {
        Deck deck = resolveOrNull(spec);
        if (deck == null) {
            System.err.println("[FlyAgent] ERROR: could not resolve deck spec '"
                    + spec + "' (not a file, not in Forge's Commander pool)");
            System.exit(2);
        }
        System.out.println("[FlyAgent] deck '" + spec + "' -> '"
                + deck.getName() + "'");
        return deck;
    }

    static Deck resolveOrNull(String spec) {
        if (spec == null || spec.isBlank()) {
            return null;
        }
        String s = spec.trim();

        // 1) Forge-randomized Commander deck from the normal UI pool
        if ("random".equalsIgnoreCase(s)) {
            return randomCommanderDeck();
        }

        // 2) explicit .dck file path (existing behaviour)
        if (s.endsWith(".dck")) {
            File f = new File(s);
            Deck fromFile = f.isFile() ? DeckSerializer.fromFile(f) : null;
            if (fromFile != null) {
                return fromFile;
            }
            // fall through: "Name.dck" may really be a pool name
            s = s.substring(0, s.length() - 4);
        } else if (s.contains(File.separator)) {
            // any other path-like spec
            Deck fromFile = DeckSerializer.fromFile(new File(s));
            if (fromFile != null) {
                return fromFile;
            }
            return null;
        }

        // 3) deck name in Forge's normal Commander storage
        IStorage<Deck> storage = commanderStorage();
        if (storage != null) {
            Deck byName = storage.get(s);
            if (byName != null) {
                return byName;
            }
            for (String name : storage.getItemNames()) {
                if (name.equalsIgnoreCase(s)) {
                    return storage.get(name);
                }
            }
        }

        // 4) name inside the full recursive DeckProxy pool (subfolders etc.)
        Deck fromPool = fromProxyPool(s);
        if (fromPool != null) {
            return fromPool;
        }
        return null;
    }

    /** Forge's own random Commander deck (the "Commander Decks" AI pool). */
    static Deck randomCommanderDeck() {
        List<DeckProxy> pool = new ArrayList<>();
        for (DeckProxy p : DeckProxy.getAllCommanderDecks()) {
            pool.add(p);
        }
        if (!pool.isEmpty()) {
            int idx = (int) Math.floor(MyRandom.getRandom().nextDouble() * pool.size());
            idx = Math.max(0, Math.min(pool.size() - 1, idx));
            Deck d = pool.get(idx).getDeck();
            if (d != null) {
                System.out.println("[FlyAgent] Forge pool has " + pool.size()
                        + " commander decks; random pick #" + idx);
                return d;
            }
        }
        // Fallback: Forge's own helper over the same storage (returns null if empty)
        return DeckgenUtil.getCommanderDeck();
    }

    /** Number of decks in the normal Commander pool (for startup info). */
    public static int poolSize() {
        int n = 0;
        for (DeckProxy p : DeckProxy.getAllCommanderDecks()) {
            n++;
        }
        return n;
    }

    private static Deck fromProxyPool(String name) {
        for (DeckProxy p : DeckProxy.getAllCommanderDecks()) {
            if (p.getName().equalsIgnoreCase(name)) {
                return p.getDeck();
            }
        }
        return null;
    }

    @SuppressWarnings("unchecked")
    private static IStorage<Deck> commanderStorage() {
        try {
            CardCollections cc = FModel.getDecks();
            if (cc == null) {
                return null;
            }
            return (IStorage<Deck>) cc.getCommander();
        } catch (Throwable t) {
            return null;
        }
    }
}
