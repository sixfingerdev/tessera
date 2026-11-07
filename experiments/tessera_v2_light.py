import torch, torch.nn as nn, torch.nn.functional as F, math

class MultiHashMemory(nn.Module):
    def __init__(self, d=64, bits=8, hashes=4):
        super().__init__()
        self.d=d; self.bits=bits; self.hashes=hashes; self.buckets=2**bits
        self.hash_projs=nn.ModuleList([nn.Linear(d,bits,bias=False) for _ in range(hashes)])
        self.bucket_mlps=nn.ModuleList([nn.Sequential(nn.Linear(d,d), nn.GELU(), nn.Linear(d,d)) for _ in range(hashes)])
        self.bucket_attn=nn.MultiheadAttention(d,2,batch_first=True)
        self.bucket_ln=nn.LayerNorm(d)
        self.gate=nn.Linear(d*hashes,d)
        self.ffn=nn.Sequential(nn.Linear(d,d*2), nn.GELU(), nn.Linear(d*2,d))
        self.ln1=nn.LayerNorm(d); self.ln2=nn.LayerNorm(d)
    def forward(self,x):
        B,N,D=x.shape
        rets=[]
        for h_idx in range(self.hashes):
            h_logits=self.hash_projs[h_idx](x)
            h_bin=(torch.sign(h_logits).clamp(min=1e-6)>0).long()
            powers=2**torch.arange(self.bits, device=x.device)
            bucket_ids=(h_bin*powers).sum(-1)%self.buckets
            bucket_sums=torch.zeros(B,self.buckets,D, device=x.device)
            bucket_counts=torch.zeros(B,self.buckets,1, device=x.device)
            bucket_sums.scatter_add_(1, bucket_ids.unsqueeze(-1).expand(-1,-1,D), x)
            bucket_counts.scatter_add_(1, bucket_ids.unsqueeze(-1), torch.ones_like(bucket_ids).unsqueeze(-1).float())
            bucket_means=bucket_sums/(bucket_counts+1e-6)
            bm_attn,_=self.bucket_attn(bucket_means,bucket_means,bucket_means)
            bucket_means=self.bucket_ln(bucket_means+bm_attn)
            bucket_means=self.bucket_mlps[h_idx](bucket_means)
            retrieved=torch.gather(bucket_means,1,bucket_ids.unsqueeze(-1).expand(-1,-1,D))
            rets.append(retrieved)
        retrieved_cat=torch.cat(rets,dim=-1)
        g=torch.sigmoid(self.gate(retrieved_cat))
        retrieved_mean=torch.stack(rets,dim=0).mean(dim=0)
        x=x+g*retrieved_mean
        x=self.ln1(x)
        x=x+self.ffn(x)
        x=self.ln2(x)
        return x

class TesseraV2(nn.Module):
    def __init__(self, vocab=64,d=64,layers=2,bits=8,hashes=4):
        super().__init__()
        self.emb=nn.Embedding(vocab,d)
        self.layers=nn.ModuleList([MultiHashMemory(d,bits,hashes) for _ in range(layers)])
        self.out=nn.Linear(d,vocab)
    def forward(self,x):
        h=self.emb(x)
        for l in self.layers: h=l(h)
        return self.out(h)

class Trans(nn.Module):
    def __init__(self,vocab=64,d=64,h=2):
        super().__init__()
        self.emb=nn.Embedding(vocab,d)
        self.attn=nn.MultiheadAttention(d,h,batch_first=True)
        self.ffn=nn.Sequential(nn.Linear(d,d*2), nn.GELU(), nn.Linear(d*2,d))
        self.ln1=nn.LayerNorm(d); self.ln2=nn.LayerNorm(d); self.out=nn.Linear(d,vocab)
    def forward(self,x):
        h=self.emb(x)
        N=x.shape[1]
        mask=torch.triu(torch.ones(N,N), diagonal=1).bool()
        h2,_=self.attn(h,h,h, attn_mask=mask)
        h=self.ln1(h+h2); h=self.ln2(h+self.ffn(h))
        return self.out(h)

def gen(vocab=64, L=128, B=32):
    x=torch.randint(0,vocab,(B,L))
    for i in range(1,L):
        m=torch.rand(B)<0.3
        x[m,i]=(x[m,i-1]+torch.randint(1,5,(m.sum(),)))%vocab
    y=torch.roll(x,-1,1)
    return x,y

def train_eval(m, L=128, steps=150):
    opt=torch.optim.AdamW(m.parameters(), lr=3e-3)
    for _ in range(steps):
        x,y=gen(L=L,B=16)
        logits=m(x)
        loss=F.cross_entropy(logits.view(-1,logits.shape[-1]), y.view(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        x,y=gen(L=L,B=64)
        logits=m(x)
        loss=F.cross_entropy(logits.view(-1,logits.shape[-1]), y.view(-1)).item()
        return loss, math.exp(loss)

print("Kalite %100 testi - Seq 128")
torch.manual_seed(0)
m1=Trans()
l1,p1=train_eval(m1,128,150)
print(f"Transformer: loss {l1:.3f} ppl {p1:.1f}")

torch.manual_seed(0)
m2=TesseraV2(layers=2,bits=8,hashes=4)
l2,p2=train_eval(m2,128,150)
print(f"Tessera V2 2L 4H: loss {l2:.3f} ppl {p2:.1f} -> {p1/p2*100:.1f}% kalite (100%+ = daha iyi)")

torch.manual_seed(0)
m3=TesseraV2(layers=2,bits=10,hashes=8)
l3,p3=train_eval(m3,128,150)
print(f"Tessera V2 2L 8H 10bit: loss {l3:.3f} ppl {p3:.1f} -> {p1/p3*100:.1f}% kalite")

print("\nContext scaling - Donanım gereksinimi")
for N in [512,2048,8192,32768,131072,1048576]:
    std_mem = N*N*4*4/1024**2
    tess_mem = (N*128*4 + 1024*128*4)/1024**2
    print(f"N={N:7d}: Transformer {std_mem:8.1f}MB {'OOM 6GB' if std_mem>6144 else ''} | Tessera {tess_mem:5.1f}MB")
