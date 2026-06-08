# Mamba2 Policy Parameters

Guide to understanding and configuring Mamba2 hyperparameters for RecurrentPPO.

## Parameter Reference

### `mamba_d_model` (default: 64)
**What:** The core hidden dimension of the Mamba2 block — the width of the representation flowing through the model. Analogous to `lstm_hidden_size`.

**Role:** This is the **bottleneck dimension**. Your observation features get projected into this space, and it's the width of the output fed to the MLP heads. Must be a **multiple of `mamba_headdim`**.

**How to choose:** Start small (64) for simple environments with low-dimensional observations. Scale up (128, 256) if your environment has complex/high-dimensional observations or long-horizon credit assignment.

---

### `mamba_d_state` (default: 64)
**What:** The dimension of the **SSM recurrent state per head** — how much information each head can carry forward across timesteps. This is the "memory capacity" of the selective state-space model.

**Role:** A larger `d_state` means each head can maintain a richer compressed summary of history. Think of it as the size of the "notebook" the model writes to and reads from at each step.

**How to choose:** 64 is already generous. The original Mamba paper uses 16–128. For RL tasks where you need to remember specific past events across many steps, increase it. For simpler reactive tasks, 16–32 may suffice.

---

### `mamba_d_conv` (default: 4)
**What:** The kernel width of the **causal 1D convolution** applied before the SSM. This is a small local convolution over the most recent timesteps.

**Role:** Captures **short-range local patterns** — essentially a sliding window of the last `d_conv` timesteps. The SSM handles long-range dependencies; the conv handles nearby context. Think of it as "how many recent frames does the model look at directly."

**How to choose:** 4 is the standard default from the Mamba papers and works well for most tasks. You rarely need to change this. Values of 2–4 are typical. Increasing it beyond 4 adds minimal benefit and increases the conv state size.

---

### `mamba_expand` (default: 2)
**What:** The **inner expansion factor**. The internal working dimension is `d_inner = d_model * expand`.

**Role:** Similar to the FFN expansion in a Transformer. With `d_model=64` and `expand=2`, the SSM internally operates at width 128, then projects back down. This gives the model more expressiveness without increasing the input/output dimension.

**How to choose:** 2 is the standard default. Increasing to 4 quadruples the computation inside the block (more like Transformer FFN ratios). For RL with small models, 2 is usually fine. Only increase if you see underfitting with a small `d_model` and can't increase `d_model` directly.

---

### `mamba_headdim` (default: 64)
**What:** The dimension **per attention-like head** within the SSM. The number of heads is derived as `nheads = d_inner / headdim`.

**Role:** Controls the **granularity of the multi-head decomposition**. With `d_model=64, expand=2`, `d_inner=128`. If `headdim=64`, you get 2 heads. If `headdim=32`, you get 4 heads. More heads = more independent "tracks" of recurrence, each with its own selective gating.

**How to choose:** With your defaults (`d_model=64, expand=2`), setting `headdim=64` gives 2 heads — a reasonable starting point. If you increase `d_model`, you may want to keep `headdim` at 64 and let the head count grow naturally. The Mamba2 paper typically uses `headdim=64`.

---

## Quick Reference

| Parameter | Controls | Increase if... | Cost of increasing |
|---|---|---|---|
| `d_model` | Representation width | Complex obs / long horizons | More params everywhere |
| `d_state` | Memory capacity per head | Need long-term memory | Larger recurrent state |
| `d_conv` | Local context window | Rarely needed | Slightly larger conv state |
| `expand` | Inner expressiveness | Underfitting, can't grow d_model | More compute per step |
| `headdim` | Head granularity | Fewer, wider heads wanted | Fewer independent heads |

---

## Typical Starting Points

**Small environment (low-dim obs, simple dynamics):**
```python
mamba_d_model=32,
mamba_d_state=32,
mamba_d_conv=4,
mamba_expand=2,
mamba_headdim=32,
```

**Default (balanced):**
```python
mamba_d_model=64,
mamba_d_state=64,
mamba_d_conv=4,
mamba_expand=2,
mamba_headdim=64,
```

**Large/complex environment (high-dim obs, long horizons):**
```python
mamba_d_model=256,
mamba_d_state=64,
mamba_d_conv=4,
mamba_expand=2,
mamba_headdim=64,
```

---

## Implementation Notes

- `d_model` must be a multiple of `headdim` (enforced via `nheads = d_inner / headdim`).
- The recurrent state size is `(nheads, headdim, d_state)` for the SSM and `(conv_dim, d_conv)` for the convolution.
- States are packed into `(h, c)` tuples for compatibility with RecurrentPPO.
