import torch, torch.nn as nn, torch.nn.functional as F, math, random
from datasets import load_dataset

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

class Hybrid(nn.Module):
    def __init__(self,vocab=5000,d=64):
        super().__init__()
        self.emb=nn.Embedding(vocab,d)
        self.l0=TransLayer(d)
        self.l1=HashMem(d,8,2)
        self.l2=HashMem(d,8,2)
        self.out=nn.Linear(d,vocab)
        self.d=d
    def forward(self,x):
        h=self.emb(x)*math.sqrt(self.d)
        h=self.l0(h); h=self.l1(h); h=self.l2(h)
        return self.out(h)

class PureTrans(nn.Module):
    def __init__(self,vocab=5000,d=64):
        super().__init__()
        self.emb=nn.Embedding(vocab,d)
        self.layers=nn.ModuleList([TransLayer(d) for _ in range(3)])
        self.out=nn.Linear(d,vocab)
        self.d=d
    def forward(self,x):
        h=self.emb(x)*math.sqrt(self.d)
        for l in self.layers: h=l(h)
        return self.out(h)

class PureTess(nn.Module):
    def __init__(self,vocab=5000,d=64):
        super().__init__()
        self.emb=nn.Embedding(vocab,d)
        self.layers=nn.ModuleList([HashMem(d,8,2) for _ in range(3)])
        self.out=nn.Linear(d,vocab)
    def forward(self,x):
        h=self.emb(x)
        for l in self.layers: h=l(h)
        return self.out(h)

from collections import Counter
print("Loading wikitext2...")
ds=load_dataset('Salesforce/wikitext','wikitext-2-raw-v1')
train_texts=[t for t in ds['train']['text'] if t.strip()!='']
val_texts=[t for t in ds['validation']['text'] if t.strip()!='']
# vocab 5000
cnt=Counter()
for t in train_texts: cnt.update(t.lower().split())
most=cnt.most_common(4996)
vocab={"<pad>":0,"<unk>":1,"<bos>":2,"<eos>":3}
for i,(w,_) in enumerate(most): vocab[w]=i+4
def encode(txt):
    ids=[vocab.get(w,1) for w in txt.lower().split()[:64]]
    ids=ids+[0]*(64-len(ids))
    return ids

def train_eval(model, steps=100):
    opt=torch.optim.AdamW(model.parameters(), lr=3e-3)
    model.train()
    for _ in range(steps):
        bt=random.sample(train_texts,16)
        xb=[]; yb=[]
        for txt in bt:
            ids=encode(txt)
            xb.append(ids[:-1]); yb.append(ids[1:])
        x=torch.tensor(xb); y=torch.tensor(yb)
        logits=model(x)
        loss=F.cross_entropy(logits.view(-1,5000), y.view(-1), ignore_index=0)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        ls=[]
        for _ in range(10):
            bt=random.sample(val_texts,16)
            xb=[]; yb=[]
            for txt in bt:
                ids=encode(txt)
                xb.append(ids[:-1]); yb.append(ids[1:])
            x=torch.tensor(xb); y=torch.tensor(yb)
            logits=model(x)
            ls.append(F.cross_entropy(logits.view(-1,5000), y.view(-1), ignore_index=0).item())
        avg=sum(ls)/len(ls)
        return avg, math.exp(avg)

print("Training 100 steps each...")
torch.manual_seed(0)
m=PureTrans()
l,p=train_eval(m,100)
print(f"Pure Trans 3L: loss {l:.3f} ppl {p:.1f}")

torch.manual_seed(0)
m=PureTess()
l,p=train_eval(m,100)
print(f"Pure Tess 3L: loss {l:.3f} ppl {p:.1f}")

torch.manual_seed(0)
m=Hybrid()
l,p=train_eval(m,100)
print(f"Hybrid 1T+2Te: loss {l:.3f} ppl {p:.1f} -> EN ÜSTÜN OLMALI")

