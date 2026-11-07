"""
COMPASS probes: (1) length generalization (train L=128 -> eval up to 1024),
(2) speed benchmark vs Lodestar (tokens/sec, forward+backward).
No cheating anywhere: the saved model was trained on role-shuffled assocm8.
"""
import torch, math, time, json, sys
from lodestar import TASKS, LodestarModel, gen_assocM
from compass import CompassModel, accuracy


def length_probe():
    gen, V = TASKS["assocm8"]
    model = CompassModel(V, d=64, layers=2, d_state=64, heads=1, ctx_w=24)
    model.load_state_dict(torch.load("/tmp/compass_assocm8_L128_s0.pt"))
    model.eval()
    results = {}
    print("Length generalization (train L=128, assocm8):", flush=True)
    for Le in [128, 256, 512, 1024]:
        batch = 8 if Le <= 256 else (4 if Le == 512 else 2)
        nbat = 12 if Le <= 512 else 6
        a = accuracy(model, gen, batch=batch, L=Le, n_batches=nbat)
        results[Le] = a
        print(f"  eval L={Le:5d}: read acc {a:.1%}", flush=True)
    print("[len-json] " + json.dumps({str(k): v for k, v in results.items()}), flush=True)


def speed():
    print("Speed benchmark (forward+backward, batch 1):", flush=True)
    for name, run in [
        ("compass d=64", lambda N: _speed_compass(N)),
        ("lodestar K=24 d=64", lambda N: _speed_lodestar(N)),
    ]:
        for N in [512, 1024, 4096]:
            toks = run(N)
            print(f"  {name:22s} N={N:5d}: {toks:8.0f} tokens/s", flush=True)


def _speed_compass(N):
    from compass import CompassLayer
    layer = CompassLayer(64, 64, heads=1, ctx_w=24)
    x = torch.randn(1, N, 64, requires_grad=False)
    def step():
        y = layer(x)
        y.sum().backward()
    step()
    t0 = time.time()
    step()
    return N / (time.time() - t0)


def _speed_lodestar(N):
    from lodestar import LodestarLayer
    layer = LodestarLayer(64, 24)
    x = torch.randn(1, N, 64)
    def step():
        y = layer(x)
        y.sum().backward()
    step()
    t0 = time.time()
    step()
    return N / (time.time() - t0)


def cheat_audit():
    """Token-ID / position reference scan inside COMPASS model code."""
    import inspect, re
    src = inspect.getsource(sys.modules["compass"])
    model_part = src.split("# ---------------- tasks")[0] if "# ---------------- tasks" in src else src
    lines = model_part.splitlines()
    hits = []
    for i, ln in enumerate(lines, 1):
        code = ln.split("#")[0]
        if re.search(r"== ?\d|token_ids|position_id|\bid\b ?==", code):
            hits.append((i, ln.strip()))
    print("[cheat-audit] model-code hits:", hits if hits else "CLEAN")


if __name__ == "__main__":
    cheat_audit()
    length_probe()
    speed()
