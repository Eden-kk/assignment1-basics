# Transformer Language Model Notes

## Common PyTorch Functions and Tensor Patterns

This note collects the PyTorch functions and tensor operations that came up repeatedly while implementing the Transformer language model in this assignment.

## `x.to(torch.float32)`

Use this when you want to upcast a tensor before a numerically sensitive operation such as squaring and taking a square root.

```python
in_dtype = x.dtype
x = x.to(torch.float32)
mean_square = torch.mean(x ** 2, dim=-1, keepdim=True)
rms = torch.sqrt(mean_square + self.eps)
return (x / rms).to(in_dtype)
```

Why this matters:

- Squaring low-precision tensors can overflow more easily.
- RMSNorm in the handout explicitly asks for an upcast to `torch.float32`.

## `torch.sqrt(...)` vs `** 0.5`

Use `** 0.5` for Python scalars and `torch.sqrt(...)` for tensors.

```python
std = (2 / (d_model + d_ff)) ** 0.5
```

```python
rms = torch.sqrt(mean_square + self.eps)
```

Rule of thumb:

- If the value is a Python number, use normal Python math.
- If the value is a tensor, use a PyTorch tensor operation.

## `torch.arange(...)`

Use `torch.arange` to create position indices or dimension indices.

### Position indices for RoPE

```python
positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
```

This creates:

```python
[0, 1, 2, ..., max_seq_len - 1]
```

Position `0` is not extra. It is required, because token positions normally start at `0`.

### Dimension-pair indices for RoPE

```python
freq_exponents = torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k
```

If `d_k = 8`, this produces:

```python
[0, 2, 4, 6] / 8
```

This has length `d_k // 2`, not `d_k + 1`. The endpoint `d_k` is excluded.

## Advanced indexing with `tensor[index_tensor]`

RoPE uses position-dependent sine and cosine values. Once those are cached, you can retrieve the right rows with advanced indexing.

```python
cos = self.cos[token_positions].to(x.dtype)
sin = self.sin[token_positions].to(x.dtype)
```

If:

```python
self.cos.shape == (max_seq_len, d_k // 2)
token_positions.shape == (batch_size, seq_len)
```

then:

```python
self.cos[token_positions].shape == (batch_size, seq_len, d_k // 2)
```

This is not broadcasting. It is indexing row `token_positions[..., i]` at each token position.

## Slicing every other element: `0::2` and `1::2`

RoPE treats pairs of features as 2D vectors. A common pattern is:

```python
x_even = x[..., 0::2]
x_odd = x[..., 1::2]
```

If:

```python
x.shape == (..., seq_len, d_k)
```

then:

```python
x_even.shape == (..., seq_len, d_k // 2)
x_odd.shape == (..., seq_len, d_k // 2)
```

This splits the last dimension into pairs:

- `x_even` gets features `0, 2, 4, ...`
- `x_odd` gets features `1, 3, 5, ...`

## `torch.stack(...)`

`torch.stack` combines tensors along a new dimension.

```python
out = torch.stack((out_even, out_odd), dim=-1)
```

If:

```python
out_even.shape == (..., seq_len, d_k // 2)
out_odd.shape == (..., seq_len, d_k // 2)
```

then:

```python
out.shape == (..., seq_len, d_k // 2, 2)
```

In RoPE, this is used to re-pair the rotated even and odd coordinates.

## `.flatten(-2)`

After stacking the rotated pairs, flatten the last two dimensions to restore the original hidden size.

```python
return torch.stack((out_even, out_odd), dim=-1).flatten(-2)
```

If:

```python
torch.stack((out_even, out_odd), dim=-1).shape == (..., seq_len, d_k // 2, 2)
```

then:

```python
.flatten(-2).shape == (..., seq_len, d_k)
```

This is a clean way to reconstruct the last dimension after pairwise rotation.

## `nn.Parameter(...)`

Use `nn.Parameter` for learnable tensors that should appear in the module's state dict and receive gradients.

### Linear weight

```python
self.weight = nn.Parameter(
    torch.empty((out_features, in_features), device=device, dtype=dtype)
)
```

### Embedding weight

```python
self.weight = nn.Parameter(
    torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
)
```

### RMSNorm weight

```python
self.weight = nn.Parameter(
    torch.ones(d_model, device=device, dtype=dtype)
)
```

Do not use `nn.Parameter` for fixed tensors such as cached RoPE sine and cosine values. Those should be buffers.

## `self.register_buffer(...)`

Use buffers for tensors that belong to the module but are not trainable.

```python
self.register_buffer("cos", torch.cos(angles), persistent=False)
self.register_buffer("sin", torch.sin(angles), persistent=False)
```

Why buffers are a good fit for RoPE:

- They move with the module across devices.
- They are not treated as trainable parameters.
- They naturally store cached values such as precomputed `cos` and `sin`.

## `load_state_dict(...)`

In the adapters, tests provide reference weights. The easiest way to inject them into your implementation is usually `load_state_dict`.

### Linear adapter

```python
linear = Linear(d_in, d_out, device=weights.device, dtype=weights.dtype)
linear.load_state_dict({"weight": weights})
return linear(in_features)
```

### Embedding adapter

```python
embedding = Embedding(vocab_size, d_model, device=weights.device, dtype=weights.dtype)
embedding.load_state_dict({"weight": weights})
return embedding(token_ids)
```

### RMSNorm adapter

```python
rmsnorm = RMSNorm(d_model, eps=eps, device=weights.device, dtype=weights.dtype)
rmsnorm.load_state_dict({"weight": weights})
return rmsnorm(in_features)
```

### SwiGLU adapter with `Linear` submodules

```python
swiglu.load_state_dict(
    {
        "w1.weight": w1_weight,
        "w2.weight": w2_weight,
        "w3.weight": w3_weight,
    }
)
```

The key names must match the structure of your module.

## `torch.nn.init.trunc_normal_(...)`

The handout uses truncated normal initialization for several components.

### Linear and SwiGLU matrices

```python
std = (2 / (d_model + d_ff)) ** 0.5
torch.nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
```

## Why `d_ff = 8/3 * d_model` for SwiGLU

In a standard Transformer FFN, the hidden dimension is often set to:

```python
d_ff = 4 * d_model
```

That choice matches the parameter count of the classic two-layer feedforward block:

- `W1`: `(d_ff, d_model)`
- `W2`: `(d_model, d_ff)`

If `d_ff = 4 * d_model`, the total parameter count is:

```text
d_model * 4d_model + 4d_model * d_model = 8 d_model^2
```

SwiGLU uses three matrices instead:

- `W1`: `(d_ff, d_model)`
- `W2`: `(d_model, d_ff)`
- `W3`: `(d_ff, d_model)`

So its total parameter count is:

```text
3 * d_model * d_ff
```

To keep the SwiGLU block at roughly the same parameter budget as a standard `4 * d_model` FFN, set:

```text
3 * d_model * d_ff ≈ 8 d_model^2
```

Solving for `d_ff` gives:

```text
d_ff ≈ 8/3 * d_model
```

That is why the handout uses `8/3 * d_model` rather than `4 * d_model` for SwiGLU.

If you used `d_ff = 4 * d_model` with SwiGLU, the block would instead have:

```text
3 * d_model * 4d_model = 12 d_model^2
```

which is substantially larger than the classic FFN.

### Embedding matrix

```python
torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)
```

## A compact RoPE example

This is the complete pattern built from the functions above.

```python
positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
freq_exponents = torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k
inv_freq = theta ** (-freq_exponents)
angles = torch.outer(positions, inv_freq)

self.register_buffer("cos", torch.cos(angles), persistent=False)
self.register_buffer("sin", torch.sin(angles), persistent=False)
```

```python
cos = self.cos[token_positions].to(x.dtype)
sin = self.sin[token_positions].to(x.dtype)

x_even = x[..., 0::2]
x_odd = x[..., 1::2]

out_even = x_even * cos - x_odd * sin
out_odd = x_even * sin + x_odd * cos

return torch.stack((out_even, out_odd), dim=-1).flatten(-2)
```

## Generation notes

Section 6 of the handout recommends a decoding function that:

- conditions on a user-provided prompt `x1...t`
- samples a continuation token by token
- stops when an `<|endoftext|>` token is produced
- supports temperature scaling and top-p sampling

The handout specifies the stopping condition clearly, but it does not force a single return convention for:

- whether the returned token sequence includes the original prompt
- whether the returned sequence includes the final `eos` token

In practice, a token-level `decode(...)` helper often returns the full sequence:

```python
prompt + generated_tokens
```

and stops immediately after sampling `eos`.

## Temperature scaling before softmax

Do not add a `temperature` argument to the generic `softmax` helper. Instead, scale the logits before calling `softmax`:

```python
next_token_logits = logits[:, -1, :]
scaled_logits = next_token_logits / temperature
probs = softmax(scaled_logits, dim=-1)
```

This keeps responsibilities clean:

- `softmax(x, dim)` stays a general tensor utility
- generation code handles temperature-specific decoding logic

## Why take `logits[:, -1, :]`

During decoding, the model outputs a next-token distribution at every sequence position:

```python
logits.shape == (batch_size, seq_len, vocab_size)
```

When generating the next token, only the final position matters:

```python
next_token_logits = logits[:, -1, :]
```

This produces:

```python
(batch_size, vocab_size)
```

which is the distribution used to sample `x_{t+1}`.

## `torch.multinomial(...)`

Use `torch.multinomial` to sample token IDs from a probability distribution.

```python
next_token = torch.multinomial(probs, num_samples=1)
```

If:

```python
probs.shape == (batch_size, vocab_size)
```

then:

```python
next_token.shape == (batch_size, 1)
```

To remove the trailing singleton dimension:

```python
next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
```

This gives:

```python
(batch_size,)
```

## Top-p sampling pattern

Given a probability distribution `q`, top-p sampling:

1. sorts token probabilities from largest to smallest
2. finds the smallest prefix whose cumulative probability is at least `p`
3. zeros out all remaining tokens
4. renormalizes
5. samples from the truncated distribution

Typical implementation pattern:

```python
sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

keep_mask = cumulative_probs <= top_p
keep_mask[..., 0] = True

shifted_keep_mask = keep_mask.clone()
shifted_keep_mask[..., 1:] = keep_mask[..., :-1]
shifted_keep_mask[..., 0] = True

filtered_sorted_probs = sorted_probs * shifted_keep_mask
filtered_probs = torch.zeros_like(probs)
filtered_probs.scatter_(-1, sorted_indices, filtered_sorted_probs)
probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
```

The `scatter_` step restores probabilities back to original vocabulary order after sorting.

## `unsqueeze(0)` for decoding

When decoding a single prompt, token IDs often start as a 1D tensor or Python list:

```python
tokens = [101, 2057, 2024]
```

Transformers expect batched token IDs:

```python
(batch_size, seq_len)
```

So convert the prompt to a batch of size 1:

```python
input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
```

This changes the shape from:

```python
(seq_len,)
```

to:

```python
(1, seq_len)
```

## `next(model.parameters()).device`

If the caller does not specify a device for decoding, a common fallback is:

```python
device = next(model.parameters()).device
```

Here:

- `model.parameters()` returns an iterator over model parameters
- `next(...)` retrieves the first parameter tensor
- `.device` reads the device that parameter lives on

This is a convenient way to place decoding tensors on the same device as the model.

## Practical summary

When writing Transformer code in this assignment:

- use `torch.arange` for positions and dimension schedules
- use `.to(torch.float32)` before numerically sensitive reductions
- use `torch.sqrt` on tensors
- use slicing like `0::2` and `1::2` for pairwise feature operations
- use `torch.stack(...).flatten(-2)` to rebuild paired dimensions
- use `nn.Parameter` for learnable weights
- use `register_buffer` for cached non-trainable tensors
- use `load_state_dict` in adapters whenever the key names line up
