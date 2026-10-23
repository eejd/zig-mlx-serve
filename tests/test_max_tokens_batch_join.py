#!/usr/bin/env python3
"""max_tokens is a hard cap when a second request's prefill lands on a stream's last token (#16).

A stream that hands out its LAST budgeted token from the serial slow path owes one more tick, the
finishing one. If a second request is admitted right after that tick, its prefill runs first (one
chunk: no decode tick inside it) and the finishing tick becomes a batched one, where a slot with no
pipeline state used to be forwarded and emit `max_tokens + 1`.

The window is one decode tick wide, so the test aims at it. Victim V (max_tokens=M) is timed solo
(D = start -> last token); trials then start V and send a >=1100-token request W at D - 22 .. D + 4 ms.
A trial is IN the window when V's stream ended more than STALL_MS after its last content delta (its
finishing tick waited out W's prefill) or V overshot. Every stream must carry at most its cap
(ignore_eos arm: V always runs to the cap). A stall also occurs when the pair is NOT batched, so it is
only a timing aid: whether the pair batches is read from the server's own log. The run first sends V
and W overlapped and requires the `[batched] batched decode engaged (slots=` engagement line (logged
once per process, the first time a batched forward runs) in `--server-log`. A batched tick is only
entered with a group of two or more slots, so the line proves a 2-slot batched tick ran (its `slots=`
is the count still live after the drain, which may be 1); without it the run FAILS, it proved nothing. It also FAILS with fewer than NEED stalled trials.

usage: test_max_tokens_batch_join.py PORT --server-log FILE [--m 40] [--trials 80] [--need 5]
exit 0 = no stream exceeded its cap; 1 = one did, the pair never batched, or the window was never reached.
"""
import http.client, json, random, statistics, sys, threading, time

PORT = int(sys.argv[1])
ARGS = dict(zip(sys.argv[2::2], sys.argv[3::2]))
SERVER_LOG = ARGS.get("--server-log")
if not SERVER_LOG:
    print("FAIL: --server-log FILE is required (the engine log is how the test knows the pair batched)")
    sys.exit(1)
M = int(ARGS.get("--m", 40))
TRIALS = int(ARGS.get("--trials", 80))
NEED = int(ARGS.get("--need", 5))
W_MAX = 8
STALL_MS = 30
rng = random.Random(16)
# W must hold at least BATCH_MIN_KV tokens: below the per-slot attention floor (1024 KV) the pad-waste cap
# splits a short stream from a long one and neither is batched (`groupKeepCount`), so the bug cannot occur.
BATCH_MIN_KV = 1100
# W must share no 3-grams with itself: a prompt that scores above the PLD gate arms prompt-lookup decoding
# for W, PLD slots decode serially, and then nothing is batched (the bug lives in the batched tick).
# Random lowercase strings tokenize into many different pieces, so their 3-gram score stays under the gate.
LETTERS = "abcdefghijklmnopqrstuvwxyz"
V_TEXT = "Count slowly from one to one hundred, one number per line."


def body(text, n, ignore_eos):
    b = {"model": "mlx-serve", "stream": True,
         "stream_options": {"include_usage": True, "continuous_usage_stats": True},
         "max_completion_tokens": n, "messages": [{"role": "user", "content": text}]}
    if ignore_eos:
        b["ignore_eos"] = True
    return b


def one(b, out):
    t0 = time.time()
    conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=300)
    conn.request("POST", "/v1/chat/completions", json.dumps(b), {"Content-Type": "application/json"})
    r = conn.getresponse()
    usage, deltas, last = None, 0, None
    while True:
        line = r.readline()
        if not line:
            break
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        p = line[5:].strip()
        if p == b"[DONE]":
            break
        j = json.loads(p)
        if j.get("usage"):
            usage = j["usage"]
        for ch in j.get("choices", []):
            if ch.get("delta", {}).get("content"):
                deltas += 1
                last = time.time()
    conn.close()
    out.update(tokens=(usage or {}).get("completion_tokens"), prompt=(usage or {}).get("prompt_tokens"), deltas=deltas, t0=t0, last=last, end=time.time())


def w_text():
    # Fresh words every call: the hot prefix cache must not shorten W's prefill.
    words = ("".join(rng.choice(LETTERS) for _ in range(rng.randint(5, 9))) for _ in range(450))
    return " ".join(words)


def trial(dt, ignore_eos):
    v, w = {}, {}
    tv = threading.Thread(target=one, args=(body(V_TEXT, M, ignore_eos), v))
    tv.start()
    time.sleep(dt / 1000)
    tw = threading.Thread(target=one, args=(body(w_text(), W_MAX, ignore_eos), w))
    tw.start()
    tv.join()
    tw.join()
    time.sleep(0.15)
    over = [(name, cap, max(r.get("tokens") or 0, r.get("deltas") or 0))
            for name, cap, r in (("victim", M, v), ("late", W_MAX, w))
            if max(r.get("tokens") or 0, r.get("deltas") or 0) > cap]
    if (w.get("prompt") or 0) < BATCH_MIN_KV:
        print("FAIL: W's prompt is %s tokens, under %d: the pair would not batch, no verdict" % (w.get("prompt"), BATCH_MIN_KV))
        sys.exit(1)
    stalled = v.get("last") is not None and (v["end"] - v["last"]) * 1000 > STALL_MS
    return stalled or bool(over), over


def pair_batches():
    """Did a batched tick run? One-shot engagement line of forwardBatchedDecode (runBatchedDecodeTick takes groups of 2+)."""
    try:
        with open(SERVER_LOG, errors="replace") as f:
            return "[batched] batched decode engaged (slots=" in f.read()
    except OSError:
        return False


def arm(ignore_eos):
    one(body("hello", 4, ignore_eos), {})
    ds = []
    for _ in range(4):
        s = {}
        one(body(V_TEXT, M, ignore_eos), s)
        ds.append((s["last"] - s["t0"]) * 1000)
        time.sleep(0.15)
    d = statistics.median(ds)
    # Overlap V and W mid-stream so two slots decode together; the engagement line must follow.
    trial(d / 2, ignore_eos)
    time.sleep(0.3)
    if not pair_batches():
        return d, -1, []
    in_window, hits = 0, []
    for _ in range(TRIALS):
        dt = max(0.0, d + rng.uniform(-22, 4))
        hot, over = trial(dt, ignore_eos)
        in_window += 1 if hot else 0
        if over:
            hits.append((round(dt), over))
            break
    return d, in_window, hits


fail = False
for ignore_eos in (True, False):
    label = "ignore_eos" if ignore_eos else "plain"
    d, in_window, hits = arm(ignore_eos)
    if hits:
        fail = True
        print("FAIL %s: a stream passed its max_tokens (V solo %.0f ms; dt ms, [who, cap, got]): %s" % (label, d, hits))
    elif in_window < 0:
        fail = True
        print("FAIL %s: no batched tick ran (no engagement line in %s: the pair decoded serially, e.g. W armed PLD or is under the 1024-KV per-slot floor); no verdict" % (label, SERVER_LOG))
    elif in_window < NEED:
        fail = True
        print("FAIL %s: only %d of %d trials stalled at the window (need %d; V solo %.0f ms); no verdict" % (label, in_window, TRIALS, NEED, d))
    else:
        print("PASS %s: %d of %d trials stalled at the window, the pair batches (V solo %.0f ms), no stream past its max_tokens" % (label, in_window, TRIALS, d))
sys.exit(1 if fail else 0)
