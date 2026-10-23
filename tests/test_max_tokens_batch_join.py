#!/usr/bin/env python3
"""max_tokens is a hard cap when a second request's prefill lands on a stream's last token (#16).

A stream that hands out its LAST budgeted token from the serial slow path owes one more tick, the
finishing one. If a second request is admitted right after that tick, its prefill runs first (one
chunk: no decode tick inside it) and the finishing tick becomes a batched one, where a slot with no
pipeline state used to be forwarded and emit `max_tokens + 1`.

The window is one decode tick wide, so the test aims at it. Victim V (max_tokens=M) is timed solo
(D = start -> last token); trials then start V and send a >=1100-token request W at D - 22 .. D + 4 ms.

What a trial proves is read per trial, from the server log (`--server-log`), never assumed:
- STALLED: V's stream ended more than STALL_MS after its last content delta. W's prefill runs in the
  same scheduler pass as the tick that follows it, and W joins `decoding` before that tick's active
  list is taken, so a stall means V's finishing tick ran with TWO active slots.
- BATCHED: no `[batched] slot serial:` line was logged during the trial. With two active slots every
  slot either batches or logs that line (`runDecodeTick` -> `noteSerial`, once per slot) when it
  decodes serial (PLD armed, grammar, logprobs, penalty, pad-waste), so none means the tick batched.
  (Holds with the group planner off, its default; `tryPlannerTick` has its own grouping.)
A trial counts toward the verdict only when it is both. An arm FAILS if a stream passes its cap, or if
fewer than NEED batched stalled trials happened within MAX_TRIALS (the pair never batched, or the window
was never hit: no verdict). Every arm runs on the same server, so each arm earns its own count.
Not proved: that the finishing tick itself was the batched one rather than the tick before it, beyond
the stall's timing; the engine has no per-tick counter (the engagement line is once per process).

usage: test_max_tokens_batch_join.py PORT --server-log FILE [--m 40] [--need 8] [--max-trials 200]
       [--w-style repetitive]   negative control: a PLD-arming W (the pair decodes serial); must FAIL
exit 0 = no stream exceeded its cap; 1 = one did, or no verdict.
"""
import http.client, json, os, random, statistics, sys, threading, time

PORT = int(sys.argv[1])
ARGS = dict(zip(sys.argv[2::2], sys.argv[3::2]))
SERVER_LOG = ARGS.get("--server-log")
if not SERVER_LOG:
    print("FAIL: --server-log FILE is required (the engine log is how the test knows the pair batched)")
    sys.exit(1)
M = int(ARGS.get("--m", 40))
NEED = int(ARGS.get("--need", 8))
MAX_TRIALS = int(ARGS.get("--max-trials", 200))
W_STYLE = ARGS.get("--w-style", "random")
W_MAX = 8
STALL_MS = 30
rng = random.Random(16)
# W must hold at least BATCH_MIN_KV tokens: below the per-slot attention floor (1024 KV) the pad-waste cap
# splits a short stream from a long one and neither is batched (`groupKeepCount`), so the bug cannot occur.
BATCH_MIN_KV = 1100
# W must share no 3-grams with itself: a prompt that scores above the PLD gate arms prompt-lookup decoding
# for W, PLD slots decode serially (`slot serial: spec_active`), and then nothing is batched.
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
    out.update(tokens=(usage or {}).get("completion_tokens"), prompt=(usage or {}).get("prompt_tokens"),
               deltas=deltas, t0=t0, last=last, end=time.time())


def w_text():
    if W_STYLE == "repetitive":
        syl = "ka to mi ru en sol ver lin dar pho zu bel nor fay quo xen yar wim gul hob".split()
        return "w%d " % rng.randrange(10**9) + " ".join("".join(rng.choice(syl) for _ in range(rng.randint(2, 3))) for _ in range(500))
    # Fresh words every call: the hot prefix cache must not shorten W's prefill.
    words = ("".join(rng.choice(LETTERS) for _ in range(rng.randint(5, 9))) for _ in range(450))
    return " ".join(words)


def log_size():
    try:
        return os.path.getsize(SERVER_LOG)
    except OSError:
        return 0


def serial_lines_since(offset):
    try:
        with open(SERVER_LOG, "rb") as f:
            f.seek(offset)
            return f.read().decode("utf-8", "replace").count("[batched] slot serial:")
    except OSError:
        return 0


def trial(dt, ignore_eos):
    """Returns (stalled, serial_lines, overshoots)."""
    off = log_size()
    v, w = {}, {}
    tv = threading.Thread(target=one, args=(body(V_TEXT, M, ignore_eos), v))
    tv.start()
    time.sleep(dt / 1000)
    tw = threading.Thread(target=one, args=(body(w_text(), W_MAX, ignore_eos), w))
    tw.start()
    tv.join()
    tw.join()
    time.sleep(0.15)
    if (w.get("prompt") or 0) < BATCH_MIN_KV:
        print("FAIL: W's prompt is %s tokens, under %d: the pair would not batch, no verdict" % (w.get("prompt"), BATCH_MIN_KV))
        sys.exit(1)
    over = [(name, cap, max(r.get("tokens") or 0, r.get("deltas") or 0))
            for name, cap, r in (("victim", M, v), ("late", W_MAX, w))
            if max(r.get("tokens") or 0, r.get("deltas") or 0) > cap]
    stalled = v.get("last") is not None and (v["end"] - v["last"]) * 1000 > STALL_MS
    return stalled, serial_lines_since(off), over


def arm(ignore_eos):
    one(body("hello", 4, ignore_eos), {})
    ds = []
    for _ in range(4):
        s = {}
        one(body(V_TEXT, M, ignore_eos), s)
        ds.append((s["last"] - s["t0"]) * 1000)
        time.sleep(0.15)
    d = statistics.median(ds)
    batched_hits, serial_hits, trials, hits = 0, 0, 0, []
    while trials < MAX_TRIALS and batched_hits < NEED:
        dt = max(0.0, d + rng.uniform(-22, 4))
        stalled, serial, over = trial(dt, ignore_eos)
        trials += 1
        if stalled and serial == 0:
            batched_hits += 1
        elif stalled:
            serial_hits += 1
        if over:
            hits.append((round(dt), over))
            break
    return d, trials, batched_hits, serial_hits, hits


fail = False
for ignore_eos in (True, False):
    label = "ignore_eos" if ignore_eos else "plain"
    d, trials, batched_hits, serial_hits, hits = arm(ignore_eos)
    if hits:
        fail = True
        print("FAIL %s: a stream passed its max_tokens (V solo %.0f ms; dt ms, [who, cap, got]): %s" % (label, d, hits))
    elif batched_hits < NEED:
        fail = True
        print("FAIL %s: only %d of %d trials were stalled AND batched (need %d; %d stalled trials logged `slot serial`, so they were not batched; V solo %.0f ms); no verdict"
              % (label, batched_hits, trials, NEED, serial_hits, d))
    else:
        print("PASS %s: %d batched stalled trials in %d (%d stalled trials were serial and not counted; V solo %.0f ms), no stream past its max_tokens"
              % (label, batched_hits, trials, serial_hits, d))
sys.exit(1 if fail else 0)
