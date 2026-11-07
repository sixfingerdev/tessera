"""
Random Position Needle Test - Claude's critique
- Gated Pointer only stores first token, so it cheats on fixed-position needle (always pos 0)
- Real test: needle at RANDOM position p in [0, N-2], last token 2 is query
- Gated Pointer should FAIL random position (only carries pos 0)
- CDPM (Content-Dependent Pointer Memory) should SUCCEED random position with O(N) for last token

Task:
- Input: [B, N], vocab 10
- For each batch: random pos p in [0, N-2], needle = 0/1 at p, last token = 2 (query), others = 2..9 random
- Label: y_last = needle (0/1)
- Only last loss

Models:
- Gated Pointer: G = X[:,0,:] only, so FAIL random (only pos 0)
- CDPM: Content-based argmax retrieval, O(N) for last token, should succeed random
- Transformer: 100% expected
- Tessera: 50.7% FAIL expected
"""

import torch, torch.nn as nn, torch.nn.functional as F, math, random, os, json, sys

class FirstTokenGatedMemoryLayer(nn.Module):
    """Gated Pointer that ONLY stores first token - cheats on fixed pos 0, FAILS random"""
    def __init__(self, d=64):
        super().__init__()
        self.gate_mlp = nn.Sequential(nn.Linear(d,d), nn.GELU(), nn.Linear(d,1), nn.Sigmoid())
        self.ffn=nn.Sequential(nn.Linear(d,d*2), nn.GELU(), nn.Linear(d*2,d))
        self.ln1=nn.LayerNorm(d); self.ln2=nn.LayerNorm(d)
    def forward(self, x):
        B,N,D = x.shape
        G = x[:,0:1,:].expand(-1,N,-1)
        gate = self.gate_mlp(x)
        Y = gate * G + (1-gate) * x
        x = self.ln1(x+Y)
        x = self.ln2(x+self.ffn(x))
        return x

class CDPM_Layer(nn.Module):
    """
    Content-Dependent Pointer Memory - GENERAL exact retrieval, O(N) for last token
    - Memory bank: M_j = X_j verbatim O(N)
    - Query: Q_last = W_q * X_last, Keys: K_j = W_k * X_j
    - Scores: S_j = Q_last · K_j for j=0..N-1, O(N) dot products
    - Retrieve: idx = argmax(S_j), Y_last = M[idx]
    - For training, use soft version: Y = Σ softmax(S_j) * V_j, differentiable, O(N)
    - For eval, hard argmax exact
    - Complexity: O(N) for last token only (needle task), O(N^2) if all tokens query (general LM)
    - To keep O(N) total for general LM, need binary tree O(N log N) or LSH O(N) approximate (loses exact)
    - For needle random position, O(N) for last token is enough and exact
    """
    def __init__(self, d=64):
        super().__init__()
        self.W_q = nn.Linear(d, d, bias=False)
        self.W_k = nn.Linear(d, d, bias=False)
        self.W_v = nn.Linear(d, d, bias=False)
        self.ffn=nn.Sequential(nn.Linear(d,d*2), nn.GELU(), nn.Linear(d*2,d))
        self.ln1=nn.LayerNorm(d); self.ln2=nn.LayerNorm(d)
        self.d=d
    def forward(self, x):
        B,N,D = x.shape
        Q = self.W_q(x)  # [B,N,D]
        K = self.W_k(x)  # [B,N,D]
        V = self.W_v(x)  # [B,N,D]

        if self.training:
            # Soft retrieval for differentiability, O(N^2) if done for all positions, but we only need last token for needle
            # For efficiency, compute retrieval for all positions but with O(N^2) - okay for N=1024
            # Scores: [B,N,N] = Q @ K^T / sqrt(D)
            scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(D)  # [B,N,N]
            # Causal mask: each position i can only retrieve from j <= i
            mask = torch.triu(torch.ones(N,N, device=x.device), diagonal=1).bool()  # upper triangular
            scores = scores.masked_fill(mask.unsqueeze(0), float('-inf'))
            # Softmax over keys
            attn = F.softmax(scores, dim=-1)  # [B,N,N]
            Y = torch.matmul(attn, V)  # [B,N,D]
        else:
            # Hard argmax for exact eval, O(N) per query if only last token, O(N^2) for all
            # For needle, we only care about last token, but compute for all for simplicity
            scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(D)  # [B,N,N]
            mask = torch.triu(torch.ones(N,N, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(mask.unsqueeze(0), float('-inf'))
            idx = scores.argmax(dim=-1)  # [B,N]
            # Gather
            idx_expanded = idx.unsqueeze(-1).expand(-1, -1, D)
            Y = torch.gather(V, 1, idx_expanded)

        x = self.ln1(x + Y)
        x = self.ln2(x + self.ffn(x))
        return x

class TransLayer(nn.Module):
    def __init__(self,d=64):
        super().__init__()
        self.attn=nn.MultiheadAttention(d,2,batch_first=True)
        self.ffn=nn.Sequential(nn.Linear(d,d*2), nn.GELU(), nn.Linear(d*2,d))
        self.ln1=nn.LayerNorm(d); self.ln2=nn.LayerNorm(d)
    def forward(self,x):
        N=x.shape[1]
        mask=torch.triu(torch.ones(N,N, device=x.device),1).bool()
        h2,_=self.attn(x,x,x, attn_mask=mask)
        x=self.ln1(x+h2)
        x=self.ln2(x+self.ffn(x))
        return x

class HashMem(nn.Module):
    def __init__(self,d=64,bits=8,hashes=2):
        super().__init__()
        self.d=d; self.bits=bits; self.hashes=hashes; self.buckets=2**bits
        self.projs=nn.ModuleList([nn.Linear(d,bits,bias=False) for _ in range(hashes)])
        self.mlps=nn.ModuleList([nn.Linear(d,d) for _ in range(hashes)])
        self.gate=nn.Linear(d*hashes,d)
        self.ffn=nn.Sequential(nn.Linear(d,d*2), nn.GELU(), nn.Linear(d*2,d))
        self.ln1=nn.LayerNorm(d); self.ln2=nn.LayerNorm(d)
    def forward(self,x):
        B,N,D=x.shape
        rets=[]
        for i in range(self.hashes):
            h=(torch.sign(self.projs[i](x)).clamp(min=1e-6)>0).long()
            p=2**torch.arange(self.bits, device=x.device)
            bid=(h*p).sum(-1)%self.buckets
            sums=torch.zeros(B,self.buckets,D, device=x.device)
            cnts=torch.zeros(B,self.buckets,1, device=x.device)
            sums.scatter_add_(1,bid.unsqueeze(-1).expand(-1,-1,D),x)
            cnts.scatter_add_(1,bid.unsqueeze(-1),torch.ones_like(bid).unsqueeze(-1).float())
            means=sums/(cnts+1e-6)
            means=self.mlps[i](means)
            ret=torch.gather(means,1,bid.unsqueeze(-1).expand(-1,-1,D))
            rets.append(ret)
        cat=torch.cat(rets,-1)
        g=torch.sigmoid(self.gate(cat))
        mean=torch.stack(rets).mean(0)
        x=self.ln1(x+g*mean)
        x=self.ln2(x+self.ffn(x))
        return x

class Model(nn.Module):
    def __init__(self, vocab=10, d=64, layers=3, model_type="transformer"):
        super().__init__()
        self.emb=nn.Embedding(vocab,d)
        self.d=d
        self.layers=nn.ModuleList()
        for _ in range(layers):
            if model_type=="transformer":
                self.layers.append(TransLayer(d))
            elif model_type=="tessera":
                self.layers.append(HashMem(d,8,2))
            elif model_type=="gated_pointer":
                self.layers.append(FirstTokenGatedMemoryLayer(d))
            elif model_type=="cdpm":
                self.layers.append(CDPM_Layer(d))
            else:
                raise ValueError(model_type)
        self.out=nn.Linear(d,vocab)
    def forward(self, x):
        h=self.emb(x)*math.sqrt(self.d)
        for l in self.layers:
            h=l(h)
        return self.out(h)

def gen_data_fixed(batch=16, seq_len=512):
    """Original fixed position needle at pos 0"""
    if seq_len>=1024: batch=8
    if seq_len>=2048: batch=4
    if seq_len>=4096: batch=2
    x=torch.randint(2,10,(batch, seq_len))
    y=torch.full((batch, seq_len), -100, dtype=torch.long)
    for b in range(batch):
        needle=random.randint(0,1)
        x[b,0]=needle
        x[b,-1]=2
        y[b,-1]=needle
    return x,y

def gen_data_random(batch=16, seq_len=512):
    """Random position needle - REAL test, not cheating"""
    if seq_len>=1024: batch=8
    if seq_len>=2048: batch=4
    if seq_len>=4096: batch=2
    x=torch.randint(2,10,(batch, seq_len))
    y=torch.full((batch, seq_len), -100, dtype=torch.long)
    for b in range(batch):
        needle=random.randint(0,1)
        pos=random.randint(0, seq_len-2)  # random position, not always 0
        x[b,pos]=needle
        x[b,-1]=2
        y[b,-1]=needle
        # Also store pos for debugging? Not needed, model must find via content (0/1 is unique)
    return x,y

def train_seed(model_type, seq_len, data_mode, seed=0, total_steps=1000, steps_per_run=200):
    if seq_len>=512: steps_per_run=100
    if seq_len>=1024: steps_per_run=20
    if seq_len>=2048: steps_per_run=10
    if seq_len>=4096: steps_per_run=5

    torch.manual_seed(seed); random.seed(seed)
    model=Model(vocab=10,d=64,layers=3,model_type=model_type)
    params=sum(p.numel() for p in model.parameters())
    print(f"{model_type} seq_len {seq_len} data {data_mode} params {params}")

    opt=torch.optim.AdamW(model.parameters(), lr=3e-4)
    ckpt_dir=f"/home/user/checkpoints_needle_{model_type}_{data_mode}_{seq_len}_seed{seed}"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path=os.path.join(ckpt_dir,"ckpt.pt"); log_path=os.path.join(ckpt_dir,"log.json")
    start_step=0; logs=[]
    if os.path.exists(ckpt_path):
        ckpt=torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt['model']); opt.load_state_dict(ckpt['opt']); start_step=ckpt['step']
        if os.path.exists(log_path):
            with open(log_path,'r') as f: logs=json.load(f)
        print(f"Resumed {model_type} {data_mode} {seq_len} from {start_step}")

    model.train()
    gen_fn = gen_data_fixed if data_mode=="fixed" else gen_data_random

    for step in range(start_step, total_steps):
        if step%50==0 and step>start_step:
            torch.save({"model":model.state_dict(),"opt":opt.state_dict(),"step":step}, ckpt_path)
            with open(log_path,'w') as f: json.dump(logs,f)
        xb,yb=gen_fn(batch=16, seq_len=seq_len)
        logits=model(xb)
        loss=F.cross_entropy(logits[:,-1,:], yb[:,-1])
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if step%20==0:
            print(f"{model_type} {data_mode} seq_len {seq_len} step {step} loss {loss.item():.4f}")
            logs.append({"step":step,"loss":loss.item()})
        if step>=start_step+steps_per_run-1:
            print(f"Reached {steps_per_run} steps, checkpoint and exit")
            break

    torch.save({"model":model.state_dict(),"opt":opt.state_dict(),"step":step+1}, ckpt_path)
    with open(log_path,'w') as f: json.dump(logs,f)

    model.eval()
    with torch.no_grad():
        correct=0; total=0
        for _ in range(50):
            xb,yb=gen_fn(batch=32, seq_len=seq_len)
            logits=model(xb)
            pred=logits[:,-1,:].argmax(-1); true=yb[:,-1]
            correct+=(pred==true).sum().item(); total+=true.numel()
        acc=correct/total
        print(f"{model_type} {data_mode} FINAL seq_len {seq_len} acc {acc:.1%} (random 50%)")
        return acc

if __name__=="__main__":
    model_type=sys.argv[1] if len(sys.argv)>1 else "cdpm"
    seq_len=int(sys.argv[2]) if len(sys.argv)>2 else 1024
    data_mode=sys.argv[3] if len(sys.argv)>3 else "random"
    seed=int(sys.argv[4]) if len(sys.argv)>4 else 0
    total=int(sys.argv[5]) if len(sys.argv)>5 else 1000
    train_seed(model_type, seq_len, data_mode, seed, total_steps=total)
