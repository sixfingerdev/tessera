"""
COMPASS - Causal Outer-product Memory with Prediction-error ASSociative writes
(Kapasite, rotasyon sayisindan (K) değil, durum geometrisinden (D) gelir)

THE PAIN THAT MOTIVATED THIS (user, round 3):
  Lodestar's capacity = K routing slots. General LM needs live associations ~ vocab
  scale -> K = Theta(live) -> speed dies. Exact + subquadratic + general stays closed.
DIFFERENT ANGLE (this file):
  Give up EXACTNESS, keep SPEED. A fixed D_k x D_v outer-product state with
  prediction-error (delta-rule) writes holds up to ~D live associations with HIGH
  accuracy (not exact). Reads are ONE matvec (O(1) per query, every position).
  Cost: O(N * D_k * D_v) per layer, CONSTANT memory — K never grows.
  Target form of the claim (honest, falsifiable):
    "Cheat-free multi-position associative retrieval >= 95% at load M <= c * D_state"
  — measured, with role-shuffled tokens, no token-ID/position info in model code.

MECHANISM (per layer, heads=1 shown):
  router ctx:   r_t = x_t + Wc(mean of last w tokens)            (binding: value must
                                                                  know its marker)
  write key:    k_t = normalize(W_k(r_t))                        (unit sphere: crosstalk
                                                                  ~ 1/sqrt(D_k))
  write value:  v_t = W_v(x_t)
  write gate:   g_t = sigmoid(W_g(r_t))                          (learned: fillers/queries
                                                                  should not write)
  delta write:  pred_t = k_t^T S_{t-1}                            (what THIS key already
                                                                  predicts)
                S_t = S_{t-1} + g_t * beta * k_t (v_t - pred_t)^T  (replaces same-key,
                                                                  Gram-Schmidt-like)
  read (EVERY position, before that position's write):
                y_t = q_t^T S_{t-1},  q_t = W_q(r_t)
  cleanup read: q'_t = q_t + Wr(y_t); y_t = q'^T S_{t-1}          (2-step Hopfield-style
                                                                  cleanup, optional)
  then: x = LN(x + gate_out * W_o(y));  x = LN(x + FFN(x))

HONEST PRIOR ART (not claimed as new physics):
  Delta rule = DeltaNet / fast weight programmers (Schmidhuber '92, Schlag '21,
  Yang et al.); cleanup reads = modern Hopfield; outer-product state = linear
  attention. What is measured here and reported honestly: the CAPACITY-per-state
  curve of the combined recipe (delta writes + unit keys + learned write gate +
  cleanup read) on CHEAT-FREE multi-position associative retrieval, i.e. the
  "95%+ general regime" the user asked for, with speed numbers.
"""

import torch, torch.nn as nn, torch.nn.functional as F, math, random, os, sys, json, time
from lodestar import TASKS


class CompassLayer(nn.Module):
    def __init__(self, d=64, d_state=64, heads=1, ctx_w=24, ffn_mult=2):
        super().__init__()
        self.d = d
        self.d_state = d_state
        self.heads = heads
        self.ctx_w = ctx_w
        self.Wc = nn.Linear(d, d)
        nn.init.uniform_(self.Wc.weight, -0.35, 0.35)   # ctx path must be able to
                                                        # DOMINATE the key (binding)
        nn.init.zeros_(self.Wc.bias)
        self.pool_u = nn.Parameter(torch.randn(d) / math.sqrt(d))  # salience for ctx pool
        # recency bias per offset: the BINDING marker is the NEAREST salient token,
        # not any salient token (multi-pair sequences have several markers in window)
        self.pool_b = nn.Parameter(-0.25 * torch.arange(24, dtype=torch.float32))
        # optional pool-balance mode (facts): soften recency so the key mixes
        # (name,object) rather than object-only — set via set_pool_balance()
        self._pb = -0.25
        # WRITE-key self-exclusion (facts fix): the value's pool must EXCLUDE the
        # value itself (attr) so the key = mix(name, object); the READ query pool
        # INCLUDES self (the object at the query position) -> query = mix(object, name).
        # Then k and q are the SAME function of (name,object) -> same-name facts get
        # DISTINCT keys (object differs) -> no delta-rule collision.
        self.k_noself = False
        # FIXED PRODUCT BINDING (facts fix v3): k = norm(RA·PA ⊙ RB·PB) where PA/PB
        # are role-salience pools (PA seeks the OBJECT-type token, PB the NAME-type
        # token; learned salience uA/uB over the ctx window), RA/RB are FIXED random
        # orthogonal projections (buffers, not learned). Exact meet-in-the-middle:
        # the query pair (name,object) sees the same tokens -> q == k for the same
        # pair; different pairs (sharing one token) stay near-orthogonal because
        # the OTHER component decorrelates: bind(n,o1).bind(n,o2) ~ cos(o1,o2) ~ 0.
        # No learning needed for separation -> addresses the learned-mixing plateau.
        self.k_prod = False
        self.register_buffer("RA", torch.zeros(d, d))
        self.register_buffer("RB", torch.zeros(d, d))
        self.uA = nn.Parameter(torch.randn(d) / math.sqrt(d))  # object-seeker
        self.uB = nn.Parameter(torch.randn(d) / math.sqrt(d))  # name-seeker
        # structural recency (fixed): for each role the NEAREST instance is the
        # right one (fact triple: obj@1,name@2; query pair: obj@0,name@1)
        self.register_buffer("prod_rb", -0.7 * torch.arange(24, dtype=torch.float32))
        self.prod_temp = nn.Parameter(torch.tensor(4.0))  # salience sharpness
        self.reset_prod_buffers()
        # HOLOGRAPHIC BINDING key (facts fix v2): diagnosis showed ALL keys collapse
        # to cos~0.92 -> addressing signal too weak (read picks right attr only ~70%).
        # Bipolar-HDC identity: bind(n,o1).bind(n,o2) = o1.o2 — the SHARED NAME
        # COMPONENT CANCELS, so keys become near-orthogonal per (name,object) pair.
        self.k_bind = False
        self.WkA = nn.Linear(d, heads * d_state, bias=False)
        self.WkB = nn.Linear(d, heads * d_state, bias=False)
        # bipolar saturation init: tanh must saturate for HDC-style cancellation
        nn.init.normal_(self.WkA.weight, std=2.0 / math.sqrt(d))
        nn.init.normal_(self.WkB.weight, std=2.0 / math.sqrt(d))
        # task-level flag (NOT content hardcode): compositional pair-queries read via
        # pooled context; bare-marker queries read clean identity. Set by train().
        self.q_pooled = False
        # NONLINEAR key composition (MLP): the pooled context mixes (name+object);
        # a per-position MLP composes pair-specific keys so same-name facts do NOT
        # share a key (prevents delta-rule overwrite collisions between
        # "Ahmet araba kirmizi" and "Ahmet bisiklet mavi").
        self.Wk = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, heads * d_state, bias=False))
        self.Wq = nn.Linear(d, heads * d_state, bias=False)
        self.Wv = nn.Linear(d, heads * d_state, bias=False)
        self.Wg = nn.Linear(d, heads)              # write gate per head
        nn.init.zeros_(self.Wg.weight)
        nn.init.constant_(self.Wg.bias, -2.5)      # gate starts ~closed: fillers must not
                                                   # flood the state; opens only where useful
        self.beta = nn.Parameter(torch.full((heads,), -1.1))  # 2*sigmoid(-1.1) ~= 0.5
        # per-head learnable state decay (gated-LA style): long filler prefixes
        # accumulate gate-leak noise; S_t = d*S_{t-1} + update cleans it.
        # init 6.0 -> decay ~= 0.9975 (long memory, mild cleaning)
        self.logit_decay = nn.Parameter(torch.full((heads,), 6.0))
        self.Wr = nn.Linear(heads * d_state, heads * d_state)  # cleanup read
        self.Wo = nn.Linear(heads * d_state, d, bias=False)
        self.gate_out = nn.Linear(d, 1)
        self.ffn = nn.Sequential(nn.Linear(d, d * ffn_mult), nn.GELU(), nn.Linear(d * ffn_mult, d))
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)

    def reset_prod_buffers(self):
        """Fixed random orthogonal binding projections (never trained)."""
        for buf in ("RA", "RB"):
            r = torch.randn(getattr(self, buf).shape)
            getattr(self, buf).copy_(torch.linalg.qr(r)[0])

    def prod_pools(self, x):
        """Role-salience pools: PA seeks the OBJECT-role token, PB the NAME-role
        token. Content-based (learned uA/uB salience over the causal window) so
        the SAME rule picks (object@1, name@2) at the write position and
        (object@0, name@1) at the read position — the offset asymmetry between
        fact-triple writes and query-pair reads disappears."""
        w = min(self.ctx_w, x.shape[1])
        B, N, D = x.shape
        pools = []
        for u in (self.uA, self.uB):
            s = (x * u).sum(-1)                                  # [B,N]
            Smat = s.unsqueeze(-1).expand(B, N, w).clone() * self.prod_temp
            Smat = Smat + self.prod_rb[:w].view(1, 1, w)         # nearest-instance bias
            pools.append(Smat)
        outs = []
        for Smat in pools:
            A = torch.softmax(Smat, dim=-1)                      # [B,N,w]
            ctx = torch.zeros_like(x)
            for o in range(w):
                if o == 0:
                    ctx = ctx + A[:, :, o:o + 1] * x
                else:
                    ctx[:, o:, :] += A[:, o:, o:o + 1] * x[:, :N - o, :]
            outs.append(ctx)
        return outs[0], outs[1]

    def prod_bind(self, PA, PB):
        """Fixed random product binding; [B,N,H*Ds] unit vectors."""
        B, N, D = PA.shape
        a = F.normalize(PA @ self.RA, dim=-1)                    # [B,N,Ds]
        b = F.normalize(PB @ self.RB, dim=-1)
        return F.normalize(a * b, dim=-1).view(B, N, -1)

    def set_pool_balance(self, b):
        self._pb = b
        self.pool_b.data.copy_(-abs(b) * torch.arange(24, dtype=torch.float32))

    def key_vectors(self, ctx):
        """Key tensor from pooled context (for diagnostics + forward)."""
        if self.k_bind:
            return F.normalize(torch.tanh(self.WkA(ctx)) * torch.tanh(self.WkB(ctx)), dim=-1)
        return self.Wk(ctx)

    def router(self, x):
        """Causal attention-pool over the last ctx_w tokens (learned salience).
        Mean-pool dilutes the binding signal (marker contributes ~1/w); a learned
        softmax pool can latch onto the salient binding token (the marker)."""
        if self.ctx_w and self.ctx_w > 1:
            w = min(self.ctx_w, x.shape[1])
            B, N, D = x.shape
            s = (x * self.pool_u).sum(-1)                       # [B,N] salience scores
            biases = (self._pb * torch.arange(w, device=x.device, dtype=x.dtype)).view(1, 1, w)
            Smat = torch.full((B, N, w), -1e4, device=x.device, dtype=x.dtype) + biases
            ctx = torch.zeros_like(x)
            for o in range(w):
                if o == 0:
                    Smat[:, :, 0] = s
                    ctx = ctx + x
                else:
                    Smat[:, o:, o] = s[:, :N - o]
                    ctx[:, o:, :] += x[:, :N - o, :]
            A = torch.softmax(Smat, dim=-1)                     # [B,N,w]
            ctx = torch.zeros_like(x)
            for o in range(w):
                if o == 0:
                    ctx = ctx + A[:, :, 0:1] * x
                else:
                    ctx[:, o:, :] += A[:, o:, o:o + 1] * x[:, :N - o, :]
            # no-self variant: renormalize attention without offset 0 (the token itself)
            A_ns = A.clone()
            A_ns[:, :, 0] = 0
            A_ns = A_ns / (A_ns.sum(-1, keepdim=True) + 1e-6)
            ctx_ns = torch.zeros_like(x)
            for o in range(1, w):
                ctx_ns[:, o:, :] += A_ns[:, o:, o:o + 1] * x[:, :N - o, :]
            r = x + self.Wc(ctx)
            return r, (ctx_ns if self.k_noself else ctx)
        else:
            return x, x

    def forward(self, x, cleanup=False):
        B, N, D = x.shape
        r, ctx = self.router(x)
        H, Ds = self.heads, self.d_state
        # WRITE key: contextual (value's key = its marker; facts: the (name,object)
        # pair's name). READ query: clean x by default (bare marker identifies itself);
        # pooled (x + ctx) for compositional pair-queries (object's nearest salient
        # token is its own name at distance 1 -> recency bias wins -> composition).
        K = self.key_vectors(ctx).view(B, N, H, Ds)
        if self.k_prod:
            # fixed product binding: roles located by content, pair bound by
            # fixed random projections; q uses the SAME binding (meet-in-middle)
            PA, PB = self.prod_pools(x)
            Kb = self.prod_bind(PA, PB).view(B, N, H, Ds)
            Q = Kb
            keys = F.normalize(Kb, dim=-1)
        else:
            if self.q_pooled:
                if self.k_bind:
                    # SAME binding for the query -> q(pair) == k(pair) (meet in the middle)
                    Qa = torch.tanh(self.WkA(r))
                    Qb = torch.tanh(self.WkB(r))
                    Q = F.normalize(Qa * Qb, dim=-1).view(B, N, H, Ds)
                else:
                    Q = self.Wq(r).view(B, N, H, Ds)
            else:
                Q = self.Wq(x).view(B, N, H, Ds)
        V = self.Wv(x).view(B, N, H, Ds)
        keys = F.normalize(K, dim=-1) if not self.k_prod else keys  # unit sphere
        g = torch.sigmoid(self.Wg(r))                       # [B,N,H] write gate
        beta = 2.0 * torch.sigmoid(self.beta)               # [H]
        outs = []
        S = torch.zeros(B, H, Ds, Ds, device=x.device, dtype=x.dtype)
        for t in range(N):
            q = Q[:, t]                                     # [B,H,Ds]
            y = torch.einsum('bhd,bhde->bhe', q, S)         # read BEFORE write
            outs.append(y)
            # write AFTER read (a query's own write cannot corrupt its own read)
            k = keys[:, t]                                  # [B,H,Ds]
            v = V[:, t]
            pred = torch.einsum('bhd,bhde->bhe', k, S)
            err = v - pred
            decay = torch.sigmoid(self.logit_decay).view(1, -1, 1, 1)
            S = S * decay
            coef = g[:, t].unsqueeze(-1) * beta.view(1, -1, 1)      # [B,H,1] explicit
            S = S + coef.unsqueeze(-1) * torch.einsum('bhd,bhe->bhde', k, err)
        y = torch.stack(outs, dim=1).reshape(B, N, H * Ds)
        if cleanup:
            # learned residual refine of the read (a true iterative Hopfield cleanup
            # against the stored state would need the state at each query position;
            # kept simple/honest: this is a post-read MLP refine)
            y = y + self.Wr(y)
        y = self.Wo(y)
        x = self.ln1(x + torch.sigmoid(self.gate_out(x)) * y)
        x = self.ln2(x + self.ffn(x))
        return x


class CompassModel(nn.Module):
    def __init__(self, vocab, d=64, layers=2, d_state=64, heads=1, ctx_w=24):
        super().__init__()
        self.d = d
        self.emb = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([CompassLayer(d, d_state, heads, ctx_w) for _ in range(layers)])
        self.out = nn.Linear(d, vocab)

    def forward(self, ids, cleanup=False):
        h = self.emb(ids) * math.sqrt(self.d)
        for l in self.layers:
            h = l(h, cleanup=cleanup)
        return self.out(h)


def accuracy(model, gen, batch, L, n_batches=20, cleanup=False):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = gen(batch, L)
            logits = model(x, cleanup=cleanup)
            if y.dim() == 2:
                mask = y != -100
                pred = logits.argmax(-1)[mask]
                correct += (pred == y[mask]).sum().item()
                total += int(mask.sum().item())
            else:
                pred = logits[:, -1, :].argmax(-1)
                correct += (pred == y).sum().item()
                total += y.numel()
    return correct / max(total, 1)


def train(task, L, seed=0, steps=1500, lr=1e-3, batch=8, d_state=64, heads=1, d=64, layers=2, eval_every=50):
    gen, vocab = TASKS[task]
    torch.manual_seed(seed)
    random.seed(seed)
    model = CompassModel(vocab, d=d, layers=layers, d_state=d_state, heads=heads, ctx_w=24)
    if task.startswith("facts"):
        for l in model.layers:
            l.q_pooled = True     # compositional (name,object) queries need pooled reads
            l.k_prod = True       # FIXED product binding (fix v3): roles by content
                                  # salience, pair bound by fixed random projections;
                                  # exact q==k for the same pair, near-orthogonal keys
                                  # across pairs -> no learned-mixing plateau
            l.set_pool_balance(0.03)   # legacy pools for the gate/router path
    # mixed-length training (L=0): pairs stay within ctx window but pair DENSITY varies;
    # this removes the train-length shortcut and makes accuracy length-robust
    L_list = [128, 256, 512] if L == 0 else None
    nparam = sum(p.numel() for p in model.parameters())
    print(f"[compass] {task} L={L} s{seed} steps={steps} d_state={d_state} heads={heads} layers={layers} params={nparam}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    t0 = time.time()
    for step in range(1, steps + 1):
        model.train()
        Lb = random.choice(L_list) if L_list else L
        x, y = gen(batch, Lb)
        logits = model(x)
        if y.dim() == 2:
            mask = y != -100
            loss = F.cross_entropy(logits[mask], y[mask])
        else:
            loss = F.cross_entropy(logits[:, -1, :], y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 25 == 0 or step == 1:
            print(f"[compass] {task} L={L} step {step} loss {loss.item():.4f} beta {2*torch.sigmoid(model.layers[0].beta).mean().item():.2f} ({time.time()-t0:.0f}s)", flush=True)
        if step % eval_every == 0:
            if L_list:
                accs = {Le: accuracy(model, gen, batch=8, L=Le, n_batches=6) for Le in L_list + [1024]}
                print(f"[compass] {task} mixed EVAL@{step} " +
                      " ".join(f"L{Le}:{a:.1%}" for Le, a in accs.items()), flush=True)
            else:
                a1 = accuracy(model, gen, batch=16, L=L, n_batches=8)
                a2 = accuracy(model, gen, batch=16, L=L, n_batches=8, cleanup=True)
                print(f"[compass] {task} L={L} EVAL@{step} read {a1:.1%} cleanup {a2:.1%}", flush=True)
    if L_list:
        final = {Le: accuracy(model, gen, batch=16, L=Le, n_batches=12) for Le in [128, 256, 512, 1024]}
        print(f"[compass] {task} mixed s{seed} FINAL " +
              " ".join(f"L{Le}:{a:.1%}" for Le, a in final.items()) + f" ({time.time()-t0:.0f}s)", flush=True)
        torch.save(model.state_dict(), f"/tmp/compass_{task}_mixed_s{seed}.pt")
        print("[compass-json] " + json.dumps({
            "task": task, "L": "mixed", "seed": seed, "steps": steps, "d_state": d_state,
            "heads": heads, "params": nparam, "acc_by_len": final,
            "wall_s": time.time() - t0}), flush=True)
        return model
    a1 = accuracy(model, gen, batch=32, L=L, n_batches=20)
    a2 = accuracy(model, gen, batch=32, L=L, n_batches=20, cleanup=True)
    print(f"[compass] {task} L={L} s{seed} FINAL read {a1:.1%} cleanup {a2:.1%} ({time.time()-t0:.0f}s)", flush=True)
    torch.save(model.state_dict(), f"/tmp/compass_{task}_L{L}_s{seed}.pt")
    print("[compass-json] " + json.dumps({
        "task": task, "L": L, "seed": seed, "steps": steps, "d_state": d_state,
        "heads": heads, "params": nparam, "acc_read": a1, "acc_cleanup": a2,
        "wall_s": time.time() - t0}), flush=True)
    return model


if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else "assocm8"
    L = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    steps = int(sys.argv[4]) if len(sys.argv) > 4 else 1500
    d_state = int(sys.argv[5]) if len(sys.argv) > 5 else 64
    heads = int(sys.argv[6]) if len(sys.argv) > 6 else 1
    batch = int(sys.argv[7]) if len(sys.argv) > 7 else 8
    train(task, L, seed=seed, steps=steps, d_state=d_state, heads=heads, batch=batch)
