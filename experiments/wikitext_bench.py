import torch, torch.nn as nn, torch.nn.functional as F, math, time, os, sys

# Tessera V2 light
class MultiHashMemory(nn.Module):
    def __init__(self, d=64, bits=8, hashes=2):
        super().__init__()
        self.d=d; self.bits=bits; self.hashes=hashes; self.buckets=2**bits
        self.hash_projs=nn.ModuleList([nn.Linear(d,bits,bias=False) for _ in range(hashes)])
        self.bucket_mlps=nn.ModuleList([nn.Sequential(nn.Linear(d,d), nn.GELU(), nn.Linear(d,d)) for _ in range(hashes)])
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

class TesseraModel(nn.Module):
    def __init__(self, vocab=10000, d=128, layers=2, bits=8, hashes=2):
        super().__init__()
        self.emb=nn.Embedding(vocab,d)
        self.layers=nn.ModuleList([MultiHashMemory(d,bits,hashes) for _ in range(layers)])
        self.out=nn.Linear(d,vocab)
    def forward(self,x):
        h=self.emb(x)
        for l in self.layers: h=l(h)
        return self.out(h)

# Wikitext-103 stats
# Train tokens: 103,227,021 (~103M)
# We'll benchmark tokens/sec on this CPU

vocab=10000
d=128
layers=2
bits=8
hashes=2
seq_len=128
batch=16

model=TesseraModel(vocab=vocab, d=d, layers=layers, bits=bits, hashes=hashes)
model.train()

# Warmup
x=torch.randint(0,vocab,(batch,seq_len))
for _ in range(5):
    logits=model(x)
    loss=logits.mean()
    loss.backward()

# Benchmark
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
print(f"BENCH: d={d} L={layers} bits={bits} hashes={hashes} seq={seq_len} batch={batch}")
print(f"Tokens: {tokens} in {elapsed:.2f}s -> {tps:.1f} tokens/sec")
print(f"Per step: {elapsed/steps:.3f}s")

# Wikitext-103: 103M tokens
wikitext_tokens=103_227_021
epochs=1
total_tokens=wikitext_tokens*epochs
est_seconds=total_tokens/tps
print(f"\nWikitext-103 {wikitext_tokens/1e6:.1f}M tokens, 1 epoch:")
print(f"  Tahmini süre: {est_seconds/3600:.2f} saat ({est_seconds/86400:.2f} gün)")
print(f"  10 epoch: {est_seconds*10/3600:.1f} saat ({est_seconds*10/86400:.1f} gün)")

# Farklı boyutlar için tahmin
for d_test,l_test in [(64,2),(128,4),(256,4)]:
    # Kabaca FLOPs ~ d^2 * layers
    factor = (d_test*d_test*l_test) / (d*d*layers)
    tps_est = tps / factor
    est = total_tokens / tps_est / 3600
    print(f"  d={d_test} L={l_test}: ~{est:.1f} saat/epoch (tahmini)")

