import torch, torch.nn as nn, time
class Trans(nn.Module):
    def __init__(self, vocab=10000, d=128, h=2):
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

vocab=10000; d=128; batch=4
model=Trans(vocab,d,2)
for L in [128,512,1024,2048,4096]:
    x=torch.randint(0,vocab,(batch,L))
    start=time.time()
    try:
        y=model(x)
        elapsed=time.time()-start
        mem = L*L*2*4/1024**2
        print(f"L={L:4d} batch={batch}: forward {elapsed:.3f}s mem~{mem:.1f}MB attention")
    except Exception as e:
        print(f"L={L} OOM {e}")

