"""
Linear Attention + Tessera/QUASAR Hybrid Needle Test
- Linear Attention O(N) not O(N^2)
- Feature map: elu+1 (Katharopoulos et al. 2020)
- Causal via cumulative sum
- Hybrid: 1 Linear Attention + 2 Tessera/QUASAR
"""
import torch, torch.nn as nn, torch.nn.functional as F, math, random, os, json, sys
sys.path.append('/home/user')

# Inline HashMem, TransLayer from previous to avoid datasets dep
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

class QuasarLayer(nn.Module):
    def __init__(self, d_model=64, dropout=0.0, num_superpositions=2):
        super().__init__()
        assert d_model % 2 == 0
        self.d_model = d_model
        self.d_c = d_model // 2
        self.num_sup = num_superpositions
        self.W_r = nn.Linear(d_model, self.d_c, bias=False)
        self.W_i = nn.Linear(d_model, self.d_c, bias=False)
        self.W_theta = nn.Linear(d_model, self.d_c * num_superpositions, bias=False)
        self.W_gate = nn.Linear(d_model, self.d_c * num_superpositions)
        self.W_out_r = nn.Linear(self.d_c * num_superpositions, self.d_c, bias=False)
        self.W_out_i = nn.Linear(self.d_c * num_superpositions, self.d_c, bias=False)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model*2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model*2, d_model),
            nn.Dropout(dropout)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        if num_superpositions > 1:
            self.sup_projs_r = nn.ModuleList([nn.Linear(d_model, self.d_c, bias=False) for _ in range(num_superpositions)])
            self.sup_projs_i = nn.ModuleList([nn.Linear(d_model, self.d_c, bias=False) for _ in range(num_superpositions)])
    def forward(self, x):
        B, N, D = x.shape
        D_c = self.d_c
        H = self.num_sup
        Xr = self.W_r(x)
        Xi = self.W_i(x)
        if H == 1:
            Sr = Xr.mean(dim=1)
            Si = Xi.mean(dim=1)
            Sr_exp = Sr.unsqueeze(1).expand(-1, N, -1)
            Si_exp = Si.unsqueeze(1).expand(-1, N, -1)
            Sr_list = [Sr_exp]
            Si_list = [Si_exp]
        else:
            Sr_list = []
            Si_list = []
            for h in range(H):
                Xr_h = self.sup_projs_r[h](x)
                Xi_h = self.sup_projs_i[h](x)
                Sr_h = Xr_h.mean(dim=1).unsqueeze(1).expand(-1, N, -1)
                Si_h = Xi_h.mean(dim=1).unsqueeze(1).expand(-1, N, -1)
                Sr_list.append(Sr_h)
                Si_list.append(Si_h)
        theta = self.W_theta(x)
        theta = theta.view(B, N, H, D_c)
        S_rot_r_list = []
        S_rot_i_list = []
        for h in range(H):
            Sr_h = Sr_list[h]
            Si_h = Si_list[h]
            th_h = theta[:, :, h, :]
            cos_th = torch.cos(th_h)
            sin_th = torch.sin(th_h)
            Rr = Sr_h * cos_th - Si_h * sin_th
            Ri = Sr_h * sin_th + Si_h * cos_th
            S_rot_r_list.append(Rr)
            S_rot_i_list.append(Ri)
        S_rot_r = torch.cat(S_rot_r_list, dim=-1)
        S_rot_i = torch.cat(S_rot_i_list, dim=-1)
        gate = torch.sigmoid(self.W_gate(x))
        S_rot_r = S_rot_r * gate
        S_rot_i = S_rot_i * gate
        Yr = self.W_out_r(S_rot_r)
        Yi = self.W_out_i(S_rot_i)
        Xr_out = Xr + Yr
        Xi_out = Xi + Yi
        y = torch.cat([Xr_out, Xi_out], dim=-1)
        y = self.ln1(x + y)
        y = self.ln2(y + self.ffn(y))
        return y

class LinearAttentionLayer(nn.Module):
    """
    Linear Attention O(N) - Katharopoulos et al. 2020
    φ(x) = elu(x) + 1
    Causal: S_i = sum_{j<=i} φ(K_j) V_j^T, Z_i = sum_{j<=i} φ(K_j)
    Output: O_i = φ(Q_i)^T S_i / (φ(Q_i)^T Z_i)
    No softmax, no N^2 matrix, just cumulative sums
    """
    def __init__(self, d=64, heads=2):
        super().__init__()
        self.d=d
        self.heads=heads
        assert d % heads == 0
        self.d_head = d // heads
        self.W_q = nn.Linear(d, d, bias=False)
        self.W_k = nn.Linear(d, d, bias=False)
        self.W_v = nn.Linear(d, d, bias=False)
        self.W_o = nn.Linear(d, d, bias=False)
        self.ffn=nn.Sequential(nn.Linear(d,d*2), nn.GELU(), nn.Linear(d*2,d))
        self.ln1=nn.LayerNorm(d); self.ln2=nn.LayerNorm(d)
    def phi(self, x):
        return F.elu(x) + 1
    def forward(self, x):
        B,N,D = x.shape
        H = self.heads
        Dh = self.d_head
        # Q,K,V: [B,N,D] -> [B,N,H,Dh]
        Q = self.W_q(x).view(B,N,H,Dh)
        K = self.W_k(x).view(B,N,H,Dh)
        V = self.W_v(x).view(B,N,H,Dh)
        # φ
        Qp = self.phi(Q)  # [B,N,H,Dh]
        Kp = self.phi(K)  # [B,N,H,Dh]
        # Causal cumulative sums: S_i = sum_{j<=i} Kp_j * V_j^T -> [B,N,H,Dh,Dh]? Actually Kp [Dh] * V [Dh] outer -> [Dh,Dh]
        # But we want efficient: S = cumsum(Kp * V) where * is outer? Let's do: S = sum Kp_j^T V_j? Wait paper:
        # For each head: S_i = sum_{j=1..i} φ(K_j) V_j^T -> [Dh, Dh]
        # Then output = φ(Q_i)^T S_i -> [Dh]
        # So we need to compute cumulative sum of outer products
        # Implementation: Kp [B,N,H,Dh,1] * V [B,N,H,1,Dh] -> [B,N,H,Dh,Dh]
        # cumsum over N
        # Then Qp [B,N,H,1,Dh] @ S [B,N,H,Dh,Dh] -> [B,N,H,1,Dh] -> [B,N,H,Dh]
        # Similarly Z_i = sum φ(K_j) -> [B,N,H,Dh], then denominator = φ(Q_i)^T Z_i -> scalar per head per token
        # For simplicity, compute per head loop? But we can vectorize
        # Compute KV outer
        Kp_exp = Kp.unsqueeze(-1)  # [B,N,H,Dh,1]
        V_exp = V.unsqueeze(-2)    # [B,N,H,1,Dh]
        KV = Kp_exp * V_exp.transpose(-1,-2)  # Actually Kp [Dh,1] * V [1,Dh] -> [Dh,Dh]? Wait V_exp [1,Dh], Kp_exp [Dh,1] -> matmul [Dh,Dh]
        # Let's do outer product: Kp[..., :, None] * V[..., None, :] -> [B,N,H,Dh,Dh]
        KV = Kp.unsqueeze(-1) * V.unsqueeze(-2)  # [B,N,H,Dh,Dh] because Kp [Dh,1] * V [1,Dh] broadcasting?
        # Check: Kp [B,N,H,Dh] -> unsqueeze(-1) => [B,N,H,Dh,1], V unsqueeze(-2) => [B,N,H,1,Dh], multiply => [B,N,H,Dh,Dh] correct
        # Cumulative sum over N
        S = torch.cumsum(KV, dim=1)  # [B,N,H,Dh,Dh]
        # Z cumulative
        Z = torch.cumsum(Kp, dim=1)  # [B,N,H,Dh]
        # Output numerator: Qp^T S -> Qp [B,N,H,Dh] dot S [B,N,H,Dh,Dh] -> [B,N,H,Dh]
        # Qp [B,N,H,1,Dh] @ S [B,N,H,Dh,Dh] -> [B,N,H,1,Dh]
        Qp_exp = Qp.unsqueeze(-2)  # [B,N,H,1,Dh]
        # S is [B,N,H,Dh,Dh], need [B,N,H,Dh,Dh] to multiply: Qp [1,Dh] @ S [Dh,Dh] -> [1,Dh]
        # So: (Qp_exp @ S) -> [B,N,H,1,Dh]
        num = torch.matmul(Qp_exp, S).squeeze(-2)  # [B,N,H,Dh]
        # Denominator: Qp^T Z -> scalar per token per head
        den = (Qp * Z).sum(-1, keepdim=True) + 1e-6  # [B,N,H,1]
        out = num / den  # [B,N,H,Dh]
        out = out.reshape(B,N,D)
        out = self.W_o(out)
        x = self.ln1(x + out)
        x = self.ln2(x + self.ffn(x))
        return x

class LinearAttentionModel(nn.Module):
    def __init__(self,vocab=10,d=64,layers=3,heads=2):
        super().__init__()
        self.emb=nn.Embedding(vocab,d)
        self.layers=nn.ModuleList([LinearAttentionLayer(d,heads) for _ in range(layers)])
        self.out=nn.Linear(d,vocab)
        self.d=d
    def forward(self,x):
        h=self.emb(x)*math.sqrt(self.d)
        for l in self.layers: h=l(h)
        return self.out(h)

class HybridLinearTesseraModel(nn.Module):
    """1 Linear Attention + 2 Tessera - O(N) total, no N^2"""
    def __init__(self,vocab=10,d=64,layers=3,trans_layers=1,heads=2,bits=8,hashes=2):
        super().__init__()
        self.emb=nn.Embedding(vocab,d)
        self.d=d
        self.layers=nn.ModuleList()
        for i in range(layers):
            if i < trans_layers:
                self.layers.append(LinearAttentionLayer(d,heads))
            else:
                self.layers.append(HashMem(d,bits,hashes))
        self.out=nn.Linear(d,vocab)
    def forward(self,x):
        h=self.emb(x)*math.sqrt(self.d)
        for l in self.layers: h=l(h)
        return self.out(h)

class HybridLinearQuasarModel(nn.Module):
    """1 Linear Attention + 2 QUASAR - O(N) total"""
    def __init__(self,vocab=10,d=64,layers=3,trans_layers=1,heads=2,num_sup=2):
        super().__init__()
        self.emb=nn.Embedding(vocab,d)
        self.d=d
        self.layers=nn.ModuleList()
        for i in range(layers):
            if i < trans_layers:
                self.layers.append(LinearAttentionLayer(d,heads))
            else:
                self.layers.append(QuasarLayer(d_model=d, dropout=0.0, num_superpositions=num_sup))
        self.out=nn.Linear(d,vocab)
    def forward(self,x):
        h=self.emb(x)*math.sqrt(self.d)
        for l in self.layers: h=l(h)
        return self.out(h)

class HybridLinearTessera2L1S(nn.Module):
    """2 Linear Attention + 1 Tessera"""
    def __init__(self,vocab=10,d=64,layers=3,heads=2,bits=8,hashes=2):
        super().__init__()
        self.emb=nn.Embedding(vocab,d)
        self.d=d
        self.layers=nn.ModuleList([
            LinearAttentionLayer(d,heads),
            LinearAttentionLayer(d,heads),
            HashMem(d,bits,hashes)
        ])
        self.out=nn.Linear(d,vocab)
    def forward(self,x):
        h=self.emb(x)*math.sqrt(self.d)
        for l in self.layers: h=l(h)
        return self.out(h)

def gen_data(batch=16, seq_len=512):
    if seq_len >= 1024:
        batch = 8
    if seq_len >= 2048:
        batch = 4
    if seq_len >= 4096:
        batch = 2
    x=torch.randint(2,10,(batch, seq_len))
    y=torch.full((batch, seq_len), -100, dtype=torch.long)
    for b in range(batch):
        needle=random.randint(0,1)
        x[b,0]=needle
        x[b,-1]=2
        y[b,-1]=needle
    return x,y

def train_seed(model_name, seq_len, seed=0, total_steps=1000, steps_per_run=200):
    if seq_len >= 512:
        steps_per_run = 100
    if seq_len >= 1024:
        steps_per_run = 20
    if seq_len >= 2048:
        steps_per_run = 10
    if seq_len >= 4096:
        steps_per_run = 5

    torch.manual_seed(seed)
    random.seed(seed)
    vocab_size=10
    d=64
    layers=3

    if model_name=="linear":
        model=LinearAttentionModel(vocab=vocab_size,d=d,layers=layers,heads=2)
    elif model_name=="hybrid_linear_tessera_1l2s":
        model=HybridLinearTesseraModel(vocab=vocab_size,d=d,layers=3,trans_layers=1,heads=2)
    elif model_name=="hybrid_linear_tessera_2l1s":
        model=HybridLinearTessera2L1S(vocab=vocab_size,d=d,layers=3)
    elif model_name=="hybrid_linear_quasar_1l2q":
        model=HybridLinearQuasarModel(vocab=vocab_size,d=d,layers=3,trans_layers=1,heads=2,num_sup=2)
    else:
        raise ValueError(f"Unknown {model_name}")

    params=sum(p.numel() for p in model.parameters())
    print(f"{model_name} seq_len {seq_len} params {params}")

    opt=torch.optim.AdamW(model.parameters(), lr=3e-4)
    ckpt_dir=f"/home/user/checkpoints_needle_{model_name}_{seq_len}_seed{seed}"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path=os.path.join(ckpt_dir,"ckpt.pt")
    log_path=os.path.join(ckpt_dir,"log.json")

    start_step=0
    logs=[]
    if os.path.exists(ckpt_path):
        ckpt=torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        opt.load_state_dict(ckpt['opt'])
        start_step=ckpt['step']
        if os.path.exists(log_path):
            with open(log_path,'r') as f:
                logs=json.load(f)
        print(f"Resumed {model_name} seq_len {seq_len} from {start_step}")

    model.train()
    for step in range(start_step, total_steps):
        if step%50==0 and step>start_step:
            torch.save({"model":model.state_dict(),"opt":opt.state_dict(),"step":step}, ckpt_path)
            with open(log_path,'w') as f:
                json.dump(logs,f)

        xb,yb=gen_data(batch=16, seq_len=seq_len)
        logits=model(xb)
        loss=F.cross_entropy(logits[:,-1,:], yb[:,-1])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step()

        if step%20==0:
            print(f"{model_name} seq_len {seq_len} step {step} loss {loss.item():.4f}")
            logs.append({"step":step,"loss":loss.item()})

        if step>=start_step+steps_per_run-1:
            print(f"Reached {steps_per_run} steps, checkpoint and exit")
            break

    torch.save({"model":model.state_dict(),"opt":opt.state_dict(),"step":step+1}, ckpt_path)
    with open(log_path,'w') as f:
        json.dump(logs,f)

    model.eval()
    with torch.no_grad():
        correct=0; total=0
        for _ in range(50):
            xb,yb=gen_data(batch=32, seq_len=seq_len)
            logits=model(xb)
            pred=logits[:,-1,:].argmax(-1)
            true=yb[:,-1]
            correct+=(pred==true).sum().item()
            total+=true.numel()
        acc=correct/total
        print(f"{model_name} FINAL seq_len {seq_len} acc {acc:.1%} (random 50%)")
        return acc

if __name__=="__main__":
    model_name=sys.argv[1] if len(sys.argv)>1 else "hybrid_linear_tessera_1l2s"
    seq_len=int(sys.argv[2]) if len(sys.argv)>2 else 1024
    seed=int(sys.argv[3]) if len(sys.argv)>3 else 0
    total=int(sys.argv[4]) if len(sys.argv)>4 else 1000
    train_seed(model_name, seq_len, seed, total_steps=total, steps_per_run=200)
