"""
Hybrid Needle test checkpointed - 1 Transformer + rest Tessera/QUASAR
No BG process, 20 steps per run for 1024 to fit 120s
"""
import torch, torch.nn as nn, torch.nn.functional as F, math, random, os, json, sys
sys.path.append('/home/user')

# Inline definitions to avoid datasets import from train_multiseed.py
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

class TransformerModel(nn.Module):
    def __init__(self,vocab=10,d=64,layers=3):
        super().__init__()
        self.emb=nn.Embedding(vocab,d)
        self.layers=nn.ModuleList([TransLayer(d) for _ in range(layers)])
        self.out=nn.Linear(d,vocab)
        self.d=d
    def forward(self,x):
        h=self.emb(x)*math.sqrt(self.d)
        for l in self.layers: h=l(h)
        return self.out(h)

class TesseraModel(nn.Module):
    def __init__(self,vocab=10,d=64,layers=3,bits=8,hashes=2):
        super().__init__()
        self.emb=nn.Embedding(vocab,d)
        self.layers=nn.ModuleList([HashMem(d,bits,hashes) for _ in range(layers)])
        self.out=nn.Linear(d,vocab)
    def forward(self,x):
        h=self.emb(x)
        for l in self.layers: h=l(h)
        return self.out(h)

from quasar import QuasarLayer

vocab_size=10
d=64
layers=3

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

class HybridTesseraModel(nn.Module):
    """1 Transformer + 2 Tessera (best PPL 216 hybrid)"""
    def __init__(self, vocab=10, d=64, layers=3, trans_layers=1, bits=8, hashes=2):
        super().__init__()
        self.emb=nn.Embedding(vocab,d)
        self.d=d
        self.layers=nn.ModuleList()
        for i in range(layers):
            if i < trans_layers:
                self.layers.append(TransLayer(d))
            else:
                self.layers.append(HashMem(d,bits,hashes))
        self.out=nn.Linear(d,vocab)
    def forward(self,x):
        h=self.emb(x)*math.sqrt(self.d) if isinstance(self.layers[0], TransLayer) else self.emb(x)
        # For simplicity, if first layer is Trans, we need sqrt scaling only for Trans? We'll do scaling for first part then keep
        # Actually Transformer scales embedding, Tessera doesn't. For hybrid, scale then proceed - scaling is okay
        # To be consistent: use scaled emb for all hybrid (Transformer style)
        # Recompute with scaling:
        # We'll just use scaled emb as Transformer does
        for l in self.layers:
            h=l(h)
        return self.out(h)

class HybridQuasarModel(nn.Module):
    """1 Transformer + 2 QUASAR"""
    def __init__(self, vocab=10, d=64, layers=3, trans_layers=1, num_sup=2):
        super().__init__()
        self.emb=nn.Embedding(vocab,d)
        self.d=d
        self.layers=nn.ModuleList()
        for i in range(layers):
            if i < trans_layers:
                self.layers.append(TransLayer(d))
            else:
                self.layers.append(QuasarLayer(d_model=d, dropout=0.0, num_superpositions=num_sup))
        self.out=nn.Linear(d,vocab)
    def forward(self,x):
        h=self.emb(x)*math.sqrt(self.d)
        for l in self.layers:
            h=l(h)
        return self.out(h)

class Hybrid2T1Model(nn.Module):
    """2 Transformer + 1 Tessera"""
    def __init__(self, vocab=10, d=64, layers=3, bits=8, hashes=2):
        super().__init__()
        self.emb=nn.Embedding(vocab,d)
        self.d=d
        self.layers=nn.ModuleList([
            TransLayer(d),
            TransLayer(d),
            HashMem(d,bits,hashes)
        ])
        self.out=nn.Linear(d,vocab)
    def forward(self,x):
        h=self.emb(x)*math.sqrt(self.d)
        for l in self.layers:
            h=l(h)
        return self.out(h)

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

    if model_name=="transformer":
        model=TransformerModel(vocab=vocab_size,d=d,layers=layers)
    elif model_name=="tessera":
        model=TesseraModel(vocab=vocab_size,d=d,layers=3,bits=8,hashes=2)
    elif model_name=="hybrid_tessera_1t2s":
        model=HybridTesseraModel(vocab=vocab_size,d=d,layers=3,trans_layers=1)
    elif model_name=="hybrid_tessera_2t1s":
        model=Hybrid2T1Model(vocab=vocab_size,d=d,layers=3)
    elif model_name=="hybrid_quasar_1t2q":
        model=HybridQuasarModel(vocab=vocab_size,d=d,layers=3,trans_layers=1,num_sup=2)
    elif model_name=="quasar":
        from quasar import QuasarModel
        model=QuasarModel(vocab=vocab_size,d_model=d,layers=layers,num_sup=2)
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

    # Test
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
    model_name=sys.argv[1] if len(sys.argv)>1 else "hybrid_tessera_1t2s"
    seq_len=int(sys.argv[2]) if len(sys.argv)>2 else 1024
    seed=int(sys.argv[3]) if len(sys.argv)>3 else 0
    total=int(sys.argv[4]) if len(sys.argv)>4 else 1000
    train_seed(model_name, seq_len, seed, total_steps=total, steps_per_run=200)
