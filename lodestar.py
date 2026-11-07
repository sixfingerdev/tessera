"""
LODESTAR - Learned Bucket-Routed Causal Associative Memory

HONEST FRAMING (read before citing anything here):
- This is NOT claimed to be a new world-shaking architecture.
- Closest prior art: product-key memory (Lample et al. 2019), Reformer LSH attention
  (Kitaev et al. 2020), Routing Transformer / clustered attention, Set Transformer
  inducing points, Linformer/Nystromformer/Luna bottlenecks, linear attention.
  The specific twist here: DECOUPLED learned write-address vs read-address routing
  into K shared slots, with causal prefix reads, evaluated on associative retrieval
  where token roles are RESHUFFLED PER SEQUENCE so no token-ID or position rule can
  be hardcoded (by the code OR by the model).
- There are NO token IDs, NO positions, NO task structure written into the model
  code. The model only sees token sequences. Whether it "cheats" is testable:
  run the vocab-permutation probe (task=needle_perm) - same code, different IDs.

MECHANISM (one layer):
  write assignment  Aw[j] = softmax(x_j W_w / tau)   [K]   where do I PUT my value
  read  assignment  Ar[i] = softmax(x_i W_r / tau)   [K]   which slot do I LOOK UP
  per-slot running stats (causal, prefix up to i):
    C[k, i] = sum_{j<=i} Aw[j,k]                (mass)
    S[k, i] = sum_{j<=i} Aw[j,k] * x_j          (value mass)
    slot[k, i] = S[k,i] / (C[k,i] + eps)        (running slot content)
  read out:  y_i = W_o ( sum_k Ar[i,k] * slot[k,i] )   -> only slots j<=i (CAUSAL)
  x = LN(x + gate*y); x = LN(x + FFN(x))

COMPLEXITY (honest):
  - per layer time O(N*K*D), memory O(N*K*D) transient (chunkable) -> subquadratic
    in N for constant K. NOT attention (no NxN), NOT a fixed first-token trick.
  - K slots = K simultaneously-exact live associations per layer (pigeonhole:
    a fixed-size state cannot losslessly hold > K independent items; needle needs 1,
    assoc2 needs 2 -> K=16 is enough HERE, and this limit is fundamental, not an
    engineering gap).
  - HARD eval (argmax assignments): exact single-slot dereference when the target
    is alone in its slot and the query routes to that slot -> exactness is
    measurable (route_exact diagnostic), NOT guaranteed by theorem.
  - For a general LM (every position queries, arbitrary content) the open problem
    STAYS OPEN: with constant K this layer is a bottleneck (lossy when > K live
    associations); scaling K with N gives O(N*K) ~ between O(N) and O(N^2), and
    worst-case exact subquadratic attention is believed impossible (quadratic-time
    completeness results). What this experiment tests is the narrower claim:
    a learned router can do CHEAT-FREE exact associative retrieval on a
    distribution, without any hardcoded IDs/positions.
"""

import torch, torch.nn as nn, torch.nn.functional as F, math, random, os, sys, json, time


class LodestarLayer(nn.Module):
    def __init__(self, d=64, slots=16, ffn_mult=2, ctx_kernel=0):
        super().__init__()
        self.d = d
        self.slots = slots
        # optional causal local context for the ROUTER only (window mean + linear mix).
        # Needed for relational binding (assoc2): a value token must sense WHICH
        # marker preceded it; roles reshuffle per sequence so token identity alone
        # does not encode the pairing. A depthwise conv cannot do cross-channel
        # marker identification, so: ctx = causal mean of last w tokens, r = x + Wc(ctx).
        # Filler-window means are ~constant across sequences -> approx-constant offset
        # for markers (harmless), while value tokens get a direction proportional to
        # their marker's embedding (w-separated) -> binding becomes linearly visible.
        # Causal (window <= i), cost O(N*w*d) trivial. Slot CONTENT still stores clean x.
        self.ctx_kernel = ctx_kernel
        if ctx_kernel > 1:
            self.ctx_w = ctx_kernel
            self.Wc = nn.Linear(d, d)
            nn.init.uniform_(self.Wc.weight, -0.05, 0.05)
            nn.init.zeros_(self.Wc.bias)
        else:
            self.conv = None
        self.Ww = nn.Linear(d, slots, bias=False)   # write address projection
        self.Wr = nn.Linear(d, slots, bias=False)   # read address projection
        self.Wo = nn.Linear(d, d, bias=False)
        self.gate = nn.Linear(d, 1)
        # learnable temperature (sharpening); clamped to sane range
        self.log_tau_w = nn.Parameter(torch.tensor(math.log(0.5)))
        self.log_tau_r = nn.Parameter(torch.tensor(math.log(0.5)))
        self.ffn = nn.Sequential(nn.Linear(d, d * ffn_mult), nn.GELU(), nn.Linear(d * ffn_mult, d))
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)

    def set_tau_init(self, tau_w=None, tau_r=None):
        if tau_w is not None:
            self.log_tau_w.data.fill_(math.log(tau_w))
        if tau_r is not None:
            self.log_tau_r.data.fill_(math.log(tau_r))

    def assignments(self, x, hard=False, ste=False):
        Aw_s, Ar_s = None, None
        if self.ctx_kernel and self.ctx_kernel > 1:
            w = self.ctx_kernel
            # exact causal window mean over [i-w+1, i], O(N*w*d)
            acc = torch.zeros_like(x)
            for off in range(w):
                if off == 0:
                    acc = acc + x
                else:
                    acc[:, off:, :] += x[:, :x.shape[1] - off, :]
            cnt = torch.clamp(torch.arange(1, x.shape[1] + 1, device=x.device, dtype=x.dtype), max=w).unsqueeze(-1)
            ctx = acc / cnt
            r = x + self.Wc(ctx)
        else:
            r = x
        tau_w = torch.exp(torch.clamp(self.log_tau_w, -2.3, 2.3)) * getattr(self, "tau_anneal", 1.0)
        tau_r = torch.exp(torch.clamp(self.log_tau_r, -2.3, 2.3))
        Aw = F.softmax(self.Ww(r) / tau_w, dim=-1)  # [B,N,K]
        Ar = F.softmax(self.Wr(r) / tau_r, dim=-1)  # [B,N,K]
        Aw_s, Ar_s = Aw, Ar
        if ste:
            # straight-through: discrete one-hot forward, soft gradient backward
            Aw = F.one_hot(Aw.argmax(-1), self.slots).float()
            Ar = F.one_hot(Ar.argmax(-1), self.slots).float()
            Aw = Aw - Aw_s.detach() + Aw_s if Aw_s is not None else Aw
            Ar = Ar - Ar_s.detach() + Ar_s if Ar_s is not None else Ar
        elif hard:
            Aw = F.one_hot(Aw.argmax(-1), self.slots).float()
            Ar = F.one_hot(Ar.argmax(-1), self.slots).float()
        return Aw, Ar


    def forward(self, x, hard=False, ste=False):
        Bc, N, D = x.shape
        Aw, Ar = self.assignments(x, hard=hard, ste=ste)  # [B,N,K]
        # causal per-slot prefix stats: position i sees only j <= i
        Xw = Aw.unsqueeze(-1) * x.unsqueeze(2)           # [B,N,K,D]
        S = Xw.cumsum(dim=1)                             # value mass  [B,N,K,D]
        C = Aw.cumsum(dim=1)                             # mass        [B,N,K]
        slot = S / (C.unsqueeze(-1) + 1e-6)              # running slot content
        y = (Ar.unsqueeze(-1) * slot).sum(2)             # [B,N,D] causal read
        y = self.Wo(y)
        g = torch.sigmoid(self.gate(x))
        x = self.ln1(x + g * y)
        x = self.ln2(x + self.ffn(x))
        return x


class LodestarModel(nn.Module):
    def __init__(self, vocab=10, d=64, layers=3, slots=16, ctx_kernel=0):
        super().__init__()
        self.d = d
        self.emb = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([LodestarLayer(d, slots, ctx_kernel=ctx_kernel) for _ in range(layers)])
        self.out = nn.Linear(d, vocab)

    def forward(self, ids, hard=False, ste=False):
        h = self.emb(ids) * math.sqrt(self.d)
        for l in self.layers:
            h = l(h, hard=hard, ste=ste)
        return self.out(h)


# ---------------- tasks (task definitions may hardcode; MODEL CODE never does) ----

def gen_needle(batch, L, needle_ids=(0, 1), query_id=2):
    """Repo-standard random-position needle: one needle token at random pos, query last."""
    fillers = [t for t in range(10) if t not in needle_ids and t != query_id]
    x = torch.tensor([[random.choice(fillers) for _ in range(L)] for _ in range(batch)])
    y = torch.full((batch,), -100, dtype=torch.long)
    for b in range(batch):
        needle = random.choice(needle_ids)
        pos = random.randint(0, L - 2)
        x[b, pos] = needle
        x[b, -1] = query_id
        y[b] = needle
    return x, y


def gen_needle_perm(batch, L):
    """VOCAB PERMUTATION PROBE: needle=5/7, query=3. Same model code path.
    If the architecture had any ID prior, this would break; if it learns, it didn't."""
    return gen_needle(batch, L, needle_ids=(5, 7), query_id=3)


def gen_assoc1(batch, L):
    """Single binding, per-sequence role shuffle (anti-hardcode diagnostic).
    vocab 12; pool {0,1,2,3}; per sequence m,v = random 2 distinct pool tokens.
    Layout: fillers, m, gap(2-20), v, fillers(>=1), q=m -> answer v.
    Value's address must come from context (its marker), not from token identity."""
    V = 12
    pool = [0, 1, 2, 3]
    fillers = list(range(4, V))
    x = torch.zeros(batch, L, dtype=torch.long)
    y = torch.full((batch,), -100, dtype=torch.long)
    for b in range(batch):
        roles = random.sample(pool, 2)
        m, v = roles
        budget = L - 4  # m, v, q + >=1 filler
        g1 = random.randint(2, min(20, max(2, budget // 6)))
        s4 = random.randint(1, min(30, max(1, budget // 6)))
        s0 = max(0, budget - g1 - s4)
        i = 0
        def put(tok, n):
            nonlocal i
            for _ in range(n):
                x[b, i] = tok if tok is not None else random.choice(fillers)
                i += 1
        put(None, s0); put(m, 1); put(None, g1); put(v, 1); put(None, s4); put(m, 1)
        y[b] = v
    return x, y


def gen_assocM(batch, L, M=4, pool_base=0, V_fixed=None):
    """Multi-binding multi-query (general-LM regime proxy): M (marker->value)
    bindings scattered in one sequence, then M QUERY POSITIONS (one per marker,
    random order) at the end. Loss/acc over ALL M query positions, not just last.
    Live distinct tokens at query time = 2M -> capacity prediction:
    K >= 2M needed for hard-exact (pigeonhole). Cheat-free: roles reshuffle."""
    V = V_fixed or (pool_base + 2 * M + 8)   # fixed vocab enables cross-M zero-shot
    pool = list(range(pool_base, pool_base + 2 * M))
    fillers = [t for t in range(V) if t not in set(pool)]
    x = torch.zeros(batch, L, dtype=torch.long)
    y = torch.full((batch, L), -100, dtype=torch.long)
    for b in range(batch):
        roles = pool[:]
        random.shuffle(roles)
        pairs = list(zip(roles[:M], roles[M:2 * M]))
        random.shuffle(pairs)
        budget = L - 3 * M   # fixed tokens: 2M (markers+values) + M (query block)
        # segments: M pair-gaps (min 2), M-1 separators (min 2), 1 pre-query (min 1), 1 prelude (min 0)
        mins = [0] + [2] * M + [2] * (M - 1) + [1]   # order: s0, gaps, seps, preq
        free = budget - sum(mins)
        assert free >= 0, (L, M)
        cuts = sorted(random.randint(0, max(free, 0)) for _ in range(len(mins) - 1))
        extras = ([cuts[0]] + [cuts[k] - cuts[k - 1] for k in range(1, len(cuts))] + [free - cuts[-1]]) if free > 0 else [0] * len(mins)
        seg = [m + e for m, e in zip(mins, extras)]
        s0 = seg[0]
        gaps = seg[1:1 + M]
        seps = seg[1 + M:1 + M + (M - 1)]
        preq = seg[-1]
        assert sum(seg) == budget, (sum(seg), budget)
        i = 0
        def put(tok, n):
            nonlocal i
            for _ in range(n):
                x[b, i] = tok if tok is not None else random.choice(fillers)
                i += 1
        put(None, s0)
        for idx, (m, v) in enumerate(pairs):
            put(m, 1); put(None, gaps[idx]); put(v, 1)
            if idx < M - 1:
                put(None, seps[idx])
        put(None, preq)
        qs = pairs[:]
        random.shuffle(qs)
        for (m, v) in qs:
            x[b, i] = m
            y[b, i] = v          # target at the query position itself
            i += 1
    return x, y


def gen_facts(batch, L, M=4):
    """Compositional fact retrieval with multi-key competition ("Ahmet'in arabasi"):
    facts = (name, object, attr) CONSECUTIVE triples; names repeat across facts with
    different objects (Ahmet araba kirmizi / Ahmet bisiklet mavi) -> keys compete;
    one bare name mention among fillers (no binding); queries at end are (name,object)
    pairs, answer = attr (loss at the object position). Roles/values reshuffled."""
    Nn, No, Na, Nf = 6, 3, 6, 6
    names = list(range(Nn)); objs = list(range(Nn, Nn + No))
    attrs = list(range(Nn + No, Nn + No + Na)); fillers = list(range(Nn + No + Na, Nn + No + Na + Nf))
    V = Nn + No + Na + Nf
    x = torch.zeros(batch, L, dtype=torch.long)
    y = torch.full((batch, L), -100, dtype=torch.long)
    for b in range(batch):
        pairs = set(); facts = []
        for _ in range(M):
            while True:
                n = random.choice(names); o = random.choice(objs)
                if (n, o) not in pairs:
                    pairs.add((n, o)); break
            facts.append((n, o, random.choice(attrs)))
        random.shuffle(facts)
        # segments: s0, fact-gaps(M-1, min2), preq(1), query-gaps(M-1, min2)
        fixed = 3 * M + 2 * M
        mins = [0] + [2] * (M - 1) + [1] + [2] * (M - 1)
        budget = L - fixed
        free = budget - sum(mins)
        assert free >= 0, (L, M)
        cuts = sorted(random.randint(0, max(free, 0)) for _ in range(len(mins) - 1))
        extras = ([cuts[0]] + [cuts[k] - cuts[k - 1] for k in range(1, len(cuts))] + [free - cuts[-1]]) if free > 0 else [0] * len(mins)
        seg = [m + e for m, e in zip(mins, extras)]
        s0, fact_gaps, preq, query_gaps = seg[0], seg[1:M], seg[M], seg[M + 1:]
        i = 0
        def put(tok, n):
            nonlocal i
            for _ in range(n):
                x[b, i] = tok if tok is not None else random.choice(fillers)
                i += 1
        put(None, s0)
        if s0 >= 3:  # distractor: bare name mention, no fact, no binding
            x[b, random.randrange(s0)] = random.choice(names)
        for gi, (n, o, a) in enumerate(facts):
            put(n, 1); put(o, 1); put(a, 1)
            if gi < M - 1:
                put(None, fact_gaps[gi])
        put(None, preq)
        qs = facts[:]; random.shuffle(qs)
        for qi, (n, o, a) in enumerate(qs):
            x[b, i] = n; x[b, i + 1] = o; y[b, i + 1] = a; i += 2
            if qi < M - 1:
                put(None, query_gaps[qi])
    return x, y


def gen_assoc2(batch, L):
    """Multi-association probe with PER-SEQUENCE ROLE SHUFFLING (anti-hardcode).
    vocab 12; pool {0,1,2,3}; per sequence a random permutation assigns
    marker1, marker2, value1, value2. Layout:
      fillers, m1, gap, v1, fillers, m2, gap, v2, fillers, q,   answer = v(q)
    Markers appear twice (definition + query); each value exactly once.
    No fixed token->role mapping exists ACROSS sequences: any ID-based rule fails.
    """
    V = 12
    pool = [0, 1, 2, 3]
    fillers = list(range(4, V))
    x = torch.zeros(batch, L, dtype=torch.long)
    y = torch.full((batch,), -100, dtype=torch.long)
    for b in range(batch):
        roles = pool[:]
        random.shuffle(roles)
        m1, m2, v1, v2 = roles
        # filler segment lengths: [pre, m1->v1, v1->m2, m2->v2, pre-q]
        # minimums: m1->v1 >= 8, m2->v2 >= 8, pre-q >= 1; sum == L-5 exactly
        budget = L - 5
        # marker->value gaps bounded to [2,20] (within the causal context window);
        # other segments take the rest, all random per sequence
        g1 = random.randint(2, min(20, max(2, budget // 8)))
        g3 = random.randint(2, min(20, max(2, budget // 8)))
        s2 = random.randint(26, min(60, max(26, budget // 3)))
        s4 = random.randint(1, min(30, max(1, budget // 8)))
        s0 = max(0, budget - g1 - s2 - g3 - s4)
        seg = [s0, g1, s2, g3, s4]
        assert sum(seg) <= budget, (seg, budget)
        i = 0
        def put(tok, n):
            nonlocal i
            for _ in range(n):
                x[b, i] = tok if tok is not None else random.choice(fillers)
                i += 1
        put(None, seg[0]); put(m1, 1); put(None, seg[1]); put(v1, 1)
        put(None, seg[2]); put(m2, 1); put(None, seg[3]); put(v2, 1)
        q = m1 if random.random() < 0.5 else m2
        put(None, seg[4]); put(q, 1)
        y[b] = v1 if q == m1 else v2
    return x, y


def gen_facts_perm(batch, L):
    """ID-BLOCK-SWAP probe for facts: roles reassigned (names <-> attrs swapped,
    objects shifted). Same code, different id->role map. Retrain-under-perm probe
    (same standard as assocm8_perm/needle_perm): if the architecture carried an
    id prior, retraining under the swap would fail."""
    import torch as _t
    x, y = gen_facts(batch, L, M=4)
    V = 21
    # blocks: names 0-5, objects 6-8, attrs 9-14, fillers 15-20
    # swap names<->attrs, rotate objects by 1, keep fillers
    remap = torch.arange(V)
    names_new, attrs_new = torch.arange(9, 15), torch.arange(0, 6)
    remap = torch.cat([attrs_new, torch.tensor([7, 8, 6]), names_new,
                       torch.arange(15, V)])
    x2 = remap[x]
    y2 = y.clone()
    y2[y >= 0] = torch.tensor(remap)[y[y >= 0]]
    return x2, y2


import functools
TASKS = {
    "needle": (gen_needle, 10),
    "needle_perm": (gen_needle_perm, 10),
    "assoc1": (gen_assoc1, 12),
    "assoc2": (gen_assoc2, 12),
    "assocm4": (functools.partial(gen_assocM, M=4), 20),
    "assocm8": (functools.partial(gen_assocM, M=8), 28),
    "assocm16": (functools.partial(gen_assocM, M=16), 44),
    "assocm8_perm": (functools.partial(gen_assocM, M=8, pool_base=5), 33),
    "facts4": (functools.partial(gen_facts, M=4), 21),
    "facts4_perm": (gen_facts_perm, 21),
    "facts8": (functools.partial(gen_facts, M=8), 21),
    "fm4": (functools.partial(gen_assocM, M=4, V_fixed=44), 44),
    "fm8": (functools.partial(gen_assocM, M=8, V_fixed=44), 44),
    "fm16": (functools.partial(gen_assocM, M=16, V_fixed=44), 44),
}


# ---------------- train / eval ----------------------------------------------------

@torch.no_grad()
def route_diagnostics(model, gen, batch, L, n_batches=10):
    """Per-layer: is the hard route exact? exact := query's hard read slot ==
    needle's hard write slot AND that slot received exactly one hard write."""
    n_layers = len(model.layers)
    exact = [0] * n_layers
    total = 0
    for _ in range(n_batches):
        x, y = gen(batch, L)
        nids = (0, 1) if gen is TASKS["needle"][0] else (5, 7)
        qid = 2 if gen is TASKS["needle"][0] else 3
        is_needle = torch.zeros_like(x, dtype=torch.bool)
        for t in nids:
            is_needle |= x == t
        is_needle[:, -1] = False
        p = is_needle.float().argmax(1)
        h = model.emb(x) * math.sqrt(model.d)
        for li, layer in enumerate(model.layers):
            Aw, Ar = layer.assignments(h, hard=True)
            kw = Aw.argmax(-1)
            kr = Ar[:, -1, :].argmax(-1)
            k_key = kw[torch.arange(len(p)), p]
            load = Aw.sum(1)[torch.arange(len(k_key)), k_key]
            exact[li] += ((k_key == kr) & (load == 1.0)).sum().item()
            # propagate h through the layer (hard) for the next layer's routing
            Xw = Aw.unsqueeze(-1) * h.unsqueeze(2)
            S = Xw.cumsum(1)
            C = Aw.cumsum(1)
            slot = S / (C.unsqueeze(-1) + 1e-6)
            yr = (Ar.unsqueeze(-1) * slot).sum(2)
            h = layer.ln1(h + torch.sigmoid(layer.gate(h)) * layer.Wo(yr))
            h = layer.ln2(h + layer.ffn(h))
        total += len(p)
    return [e / max(total, 1) for e in exact]


def evaluate(model, gen, batch, L, n_batches=25):
    """soft: everything soft. hardread: hard READ assignment, soft memory writes.
    hard: fully discrete. hardread is the 'one dereference, graded memory' mode."""
    model.eval()
    accs = {}
    with torch.no_grad():
        for mode in ("soft", "hardread", "hard"):
            correct = 0
            total = 0
            for _ in range(n_batches):
                x, y = gen(batch, L)
                if y.dim() == 2:
                    mask = y != -100
                else:
                    mask = None
                if mode == "soft":
                    logits = model(x)
                elif mode == "hardread":
                    h = model.emb(x) * math.sqrt(model.d)
                    for l in model.layers:
                        Aw, Ar = l.assignments(h)
                        k = Ar.argmax(-1)
                        Ar = F.one_hot(k, l.slots).float()
                        Xw = Aw.unsqueeze(-1) * h.unsqueeze(2)
                        S = Xw.cumsum(1)
                        C = Aw.cumsum(1)
                        slot = S / (C.unsqueeze(-1) + 1e-6)
                        yr = (Ar.unsqueeze(-1) * slot).sum(2)
                        h = l.ln1(h + torch.sigmoid(l.gate(h)) * l.Wo(yr))
                        h = l.ln2(h + l.ffn(h))
                    logits = model.out(h)
                else:
                    logits = model(x, hard=True)
                if mask is not None:
                    pred = logits.argmax(-1)[mask]
                    correct += (pred == y[mask]).sum().item()
                    total += int(mask.sum().item())
                else:
                    pred = logits[:, -1, :].argmax(-1)
                    correct += (pred == y).sum().item()
                    total += y.numel()
            accs[mode] = correct / max(total, 1)
    return accs


def train(task, L, seed=0, steps=400, lr=1e-3, batch=4, slots=16, d=64, layers=3, eval_every=50):
    gen, vocab = TASKS[task]
    ctx_kernel = 24 if task.startswith("assoc") else 0
    ste_hard = False
    torch.manual_seed(seed)
    random.seed(seed)
    model = LodestarModel(vocab=vocab, d=d, layers=layers, slots=slots, ctx_kernel=ctx_kernel)
    if task.startswith("assoc"):
        for l in model.layers:
            l.set_tau_init(tau_w=0.5, tau_r=0.1)   # sharp reads: query commits to one slot early
    nparam = sum(p.numel() for p in model.parameters())
    print(f"[lodestar] task={task} L={L} seed={seed} steps={steps} lr={lr} batch={batch} slots={slots} params={nparam}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    t0 = time.time()
    for step in range(1, steps + 1):
        model.train()
        # explicit task-agnostic annealing: after 40% of training, sharpen WRITES
        # gradually (1 -> 0.15) so soft routing discretizes before hard eval.
        if task.startswith("assoc") and step > 0.6 * steps:
            frac = (step - 0.6 * steps) / (0.4 * steps)
            anneal = max(0.25, 1.0 - 0.75 * frac)
            for l in model.layers:
                l.tau_anneal = anneal
        x, y = gen(batch, L)
        logits = model(x, ste=ste_hard)
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
        if step % 20 == 0 or step == 1:
            print(f"[lodestar] {task} L={L} s{seed} step {step} loss {loss.item():.4f} "
                  f"tau_w {math.exp(model.layers[0].log_tau_w.item()):.2f} "
                  f"tau_r {math.exp(model.layers[0].log_tau_r.item()):.2f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        if step % eval_every == 0:
            accs = evaluate(model, gen, batch=16, L=L, n_batches=10)
            print(f"[lodestar] {task} L={L} s{seed} EVAL@{step} soft {accs['soft']:.1%} hard {accs['hard']:.1%}", flush=True)
    accs = evaluate(model, gen, batch=32, L=L, n_batches=25)
    line = (f"[lodestar] {task} L={L} s{seed} FINAL soft {accs['soft']:.1%} "
            f"hardread {accs['hardread']:.1%} hard {accs['hard']:.1%}")
    out = {"task": task, "L": L, "seed": seed, "steps": steps, "lr": lr,
           "params": nparam, "soft_acc": accs["soft"], "hardread_acc": accs["hardread"],
           "hard_acc": accs["hard"], "final_loss": loss.item(), "wall_s": time.time() - t0}
    if task.startswith("needle"):
        rx = route_diagnostics(model, gen, batch=16, L=L, n_batches=10)
        line += f" route_exact_per_layer {['%.1f%%' % (100*r) for r in rx]}"
        out["route_exact_per_layer"] = rx
    print(line, flush=True)
    torch.save(model.state_dict(), f"/tmp/lode_{task}_{L}_s{seed}.pt")
    print("[lodestar-json] " + json.dumps(out), flush=True)
    return model, out


if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else "needle"
    L = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    steps = int(sys.argv[4]) if len(sys.argv) > 4 else 400
    lr = float(sys.argv[5]) if len(sys.argv) > 5 else 1e-3
    slots = int(sys.argv[6]) if len(sys.argv) > 6 else (24 if task.startswith("assoc") else 16)
    batch = int(sys.argv[7]) if len(sys.argv) > 7 else 4
    train(task, L, seed=seed, steps=steps, lr=lr, batch=batch, slots=slots)
