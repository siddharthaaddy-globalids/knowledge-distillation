# Demo Prompts — Original vs Distilled 135M

Prompts where the **base model fails and the distilled model succeeds, in 3/3 runs**.

Settings: `temperature=0.3`, `max_new_tokens=64`, `repetition_penalty=1.15`.

The difference is **instruction-shape compliance**: the base model cannot suppress a
preamble or honour an exact item count. The distilled model can.

---

### 1. `List three states of matter. Use a numbered list.`

| | Output |
|---|---|
| Original | `Here is the list of states of matter: 1. Solid (solid): A substance that remains in its original state when heated...` |
| Distilled | `1. Solid: A solid is a substance that maintains its shape and volume... 2. Liquid: ...` |

---

### 2. `Name three programming data types, formatted as a numbered list.`

| | Output |
|---|---|
| Original | `Here is the listed programming data types in a numbered format: 1. **Numbering List** - Integer (int) - Float (float) - String (string) - Boolean (bool)` — one item containing four types |
| Distilled | `1. Number: 3` / `2. String: "Hello World"` / `3. Boolean: True or False` — exactly three |

---

### 3. `What are three ways to reduce household energy use? Present them as a numbered list.`

| | Output |
|---|---|
| Original | `Here is the provided list of five ways to reduce household energy use: 1. ...` — announces **five** when asked for three |
| Distilled | `1. Switch off lights and electronics when not in use: ...` — starts at the list, gives three |

---

### 4. `List three renewable energy sources. Use a numbered list.`

| | Output |
|---|---|
| Original | `Here is the list of renewable energy sources: 1. Solar Energy (Solar Power) - This source harnesses...` |
| Distilled | `1. Solar Energy: This is the most widely used source of renewable energy... 2. Wind Energy: ...` |

---

## Do not demo these

| Prompt | Why |
|---|---|
| `Give me three planets in our solar system. Answer only with the numbered list.` | Base is **better**. Distilled adds unrequested descriptions that are factually wrong ("Mercury — 3rd planet from the Sun"). |
| Any algebra (`Solve for x: 5x + 8 = 43`) | Not reliable — 3/12 correct vs 1/12, inside sampling noise. |
| `Answer with a single word: what colour is a ripe banana?` | Distilled **regressed** — more verbose than the base, neither answers "Yellow". |
| Logic (`Anna is taller than Ben...`) | Both models fail. |
| Factual/science (`Why does ice float?`) | Distilled is fluent but wrong ("denser than water"). |
