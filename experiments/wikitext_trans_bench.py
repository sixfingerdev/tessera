import torch, torch.nn as nn, torch.nn.functional as F, math, time

class TransformerTiny(nn.Module):
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

vocab=10000
d=128
seq_len=128
batch=16

model=TransformerTiny(vocab=vocab,d=d,h=2)
model.train()
x=torch.randint(0,vocab,(batch,seq_len))
for _ in range(5):
    logits=model(x)
    loss=logits.mean()
    loss.backward()

steps=20
start=time.time()
tokens=0
for _ in range(steps):
    x=torch.randint(0,vocab,(batch,seq_len))
    y=torch.randint(0,vocab,(batch,seq_len))
    logits=model(x)
    loss=F.cross_entropy(logits.view(-1,vocab), y.view(-1))
    loss.backward()
    tokens+=batch*seq_len
elapsed=time.time()-start
tps=tokens/elapsed
print(f"Transformer BENCH: d={d} seq={seq_len} batch={batch}")
print(f"Tokens: {tokens} in {elapsed:.2f}s -> {tps:.1f} tokens/sec")
print(f"Per step: {elapsed/steps:.3f}s")
wikitext_tokens=103_227_021
est=wikitext_tokens/tps/3600
print(f"Wikitext-103 1 epoch: {est:.2f} saat, 10 epoch: {est*10:.1f} saat")

# Longer context
for L in [512, 1024, 2048]:
    try:
        x=torch.randint(0,vocab,(batch,L))
        start=time.time()
        logits=model(x)
        loss=logits.mean()
        loss.backward()
        elapsed=time.time()-start
        print(f"Seq {L}: forward+backward {elapsed:.3f}s")
    except Exception as e:
        print(f"Seq {L}: OOM or error {e}")

