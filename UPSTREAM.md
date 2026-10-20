# Upstream and adopted work (`local` branch)

Primary upstream: `ddalcu/mlx-serve` (GitHub fork parent). Secondary upstreams tracked: none.

`local` is upstream's release tag the newest `v<base>-local<NN>` tag names, plus the topic
branches below. `main` mirrors `ddalcu/mlx-serve` `main` (fast-forward only).

## Adopted from other repos

None. (Commits adopted from other repos would be listed here with `Source: <repo>@<sha>` and
`Co-authored-by` trailers on the commit.)

One commit carries a `Source:` trailer that is a reference, not adopted code: it follows a
behaviour change of MLX itself (`ml-explore/mlx@f8aaf49d4a7e`, #4516) in mlx-serve's own copies of
MLX's quantized kernels.

## Topic branches

| Topic | Upstream PR | Status |
|---|---|---|
| `topic/f32-xsum-qmv` | none yet | not-submitted; tracked in eejd/zig-mlx-serve#3. Applies unchanged to upstream `main` (`ba0088f7`) and to `v26.10.2`. Blocked for upstream on its MLX pin: `lib/mlx-src` is v0.32.3 (`64ea011c`) on `main` and at `v26.10.2`, whose qmv still sums in bf16, so upstream's own bit-equality tests would fail there until the pin moves to >= 0.32.4 (or the sum is selected by MLX version) |
| `topic/minicpm5-typed-arguments` | none yet | not-submitted; tracked in eejd/zig-mlx-serve#14. Array/object tool arguments written as Python `repr` (MiniCPM5: `['a', 'b']`, `{'k': 1}`) were shipped as strings; the schema coercion now reads a strict Python literal of the declared container kind. Touches `src/chat.zig` (+ tests in `src/server.zig`, a note in `docs/gotchas/tool-calling.md`); no file in common with `topic/f32-xsum-qmv` |

`topic/f32-xsum-qmv`: mlx-serve's own copies of MLX's qmv `load_vector` (rowqmv.zig, the
`msv_qmv_rows`, verify paired gate-up, verify expert reuse, MTP coarse-pairs and Gemma gelu-ple
kernels in transformer.zig, `glm5_qmv_header.metal`) added each run of 4 (8 at 3/5 bits)
activations in bf16, to stay bit-equal to stock `quantized_matmul`. MLX 0.32.4
(`ml-explore/mlx#4516`) sums them in float, so against MLX >= 0.32.4 those kernels are no longer
bit-equal to stock and carry about three times stock's error against an f32 reference (measured:
eejd/zig-mlx-serve#3). The topic widens each term to float. Needed by the MacPorts port
`mlx-serve-local` (macports-ports-local), which builds against `port:mlx` 0.32.4.

## Origin

Developed 2026-10-17 from the MacPorts deploy finding in macports-ports-local#1361 (the stable
`mlx-serve` 26.10.2 builds against mlx 0.32.4 only with a packaging patch, and then fails seven of
its own tests).
