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
