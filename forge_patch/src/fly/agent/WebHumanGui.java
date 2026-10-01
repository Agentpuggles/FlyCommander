package fly.agent;

import forge.gui.interfaces.IGuiGame;
import java.lang.reflect.*;
import java.util.*;

/** Partial headless IGuiGame adapter used by the digital controller test.
 * Never supplies AI answers for unsupported input. A GUI view is optional for
 * the pinned YieldController; it is not the authoritative engine GameView.
 */
public final class WebHumanGui implements InvocationHandler {
    private final DecisionBroker broker;
    public WebHumanGui(DecisionBroker broker) { this.broker = broker; }
    public IGuiGame create() {
        return (IGuiGame) Proxy.newProxyInstance(IGuiGame.class.getClassLoader(),
                new Class<?>[]{IGuiGame.class}, this);
    }
    @Override public Object invoke(Object proxy, Method method, Object[] args) throws Throwable {
        if (method.getDeclaringClass() == Object.class) {
            return switch (method.getName()) {
                case "toString" -> "MagicFlyHumanGui";
                case "hashCode" -> System.identityHashCode(proxy);
                case "equals" -> proxy == args[0];
                default -> throw new UnsupportedOperationException(method.getName());
            };
        }
        if (method.isDefault()) return InvocationHandler.invokeDefault(proxy, method, args);
        return switch (method.getName()) {
            // Forge 2.0.15 YieldController.shouldAutoYield and the human
            // skipsPromptForStackOrPhase path explicitly guard a null GUI view.
            // This is a headless view lookup, NOT a human decision or auto-pass.
            case "getGameView" -> null;
            case "confirm" -> choose("confirm", (String)args[1], List.of("Yes", "No"), 1, 1).get(0).equals("Yes");
            case "one" -> choose("choose_one", (String)args[0], (List<?>)args[1], 1, 1).get(0);
            case "oneOrNone" -> {
                List<?> picked = choose("choose_one", (String)args[0], (List<?>)args[1], 0, 1);
                yield picked.isEmpty() ? null : picked.get(0);
            }
            case "getChoices" -> choose("choose_cards", (String)args[0], (List<?>)args[3], (int)args[1], (int)args[2]);
            case "many" -> choose("choose_cards", (String)args[0], (List<?>)args[4], (int)args[2], (int)args[3]);
            default -> throw new UnsupportedOperationException("Human GUI input not implemented: " + method);
        };
    }
    private List<?> choose(String kind, String message, List<?> choices, int min, int max) throws Exception {
        // Objects stay on the game thread. IDs are scoped to the request, never card names.
        List<DecisionBroker.Option> options = new ArrayList<>();
        for (int i = 0; i < choices.size(); i++)
            options.add(new DecisionBroker.Option(Integer.toString(i), "Option " + (i + 1)));
        // Labels intentionally opaque until viewer-safe Forge view formatting is wired.
        List<String> selected = broker.ask(kind, message, options, min, Math.min(max, choices.size()), 0);
        List<Object> result = new ArrayList<>();
        for (String id : selected) result.add(choices.get(Integer.parseInt(id)));
        return result;
    }
}
