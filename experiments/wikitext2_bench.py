import torch, torch.nn as nn, torch.nn.functional as F, math, time, os, urllib.request, zipfile

# Download wikitext-2 if not exists
url="https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip"
zip_path="/tmp/wikitext-2.zip"
if not os.path.exists(zip_path):
    print("Downloading Wikitext-2...")
    try:
        urllib.request.urlretrieve(url, zip_path)
        print("Downloaded")
    except Exception as e:
        print(f"Download failed {e}, using synthetic")
        zip_path=None

if zip_path and os.path.exists(zip_path):
    import pathlib
    dest="/tmp/wikitext-2"
    if not os.path.exists(dest):
        os.makedirs(dest, exist_ok=True)
        with zipfile.ZipFile(zip_path,'r') as z:
            z.extractall("/tmp")
    # Count tokens roughly
    train_path="/tmp/wikitext-2/wiki.train.tokens"
    if os.path.exists(train_path):
        with open(train_path,'r',encoding='utf-8') as f:
            text=f.read()
            tokens=text.split()
            print(f"Wikitext-2 train tokens: {len(tokens)}")
            print(f"File size: {os.path.getsize(train_path)/1024**2:.1f} MB")
    else:
        print("train file not found, listing /tmp/wikitext-2")
        print(os.listdir("/tmp/wikitext-2") if os.path.exists("/tmp/wikitext-2") else "no dir")

# Benchmark Tessera on wikitext-2 size
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

vocab=10000
d=128
layers=2
bits=8
hashes=2
seq_len=128
batch=16

model=TesseraModel(vocab=vocab, d=d, layers=layers, bits=bits, hashes=hashes)
model.train()

x=torch.randint(0,vocab,(batch,seq_len))
for _ in range(3):
    logits=model(x)
    loss=logits.mean()
    loss.backward()

steps=30
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
print(f"\nBENCH: {tps:.1f} tokens/sec")

wikitext2_tokens=2_088_628
for epochs in [1,10]:
    total=wikitext2_tokens*epochs
    sec=total/tps
    print(f"Wikitext-2 {wikitext2_tokens/1e6:.2f}M tokens {epochs} epoch: {sec/60:.1f} dakika ({sec/3600:.2f} saat)")

# Also wikitext-103 for comparison
wikitext103=103_227_021
sec=wikitext103/tps
print(f"Wikitext-103 103M 1 epoch: {sec/3600:.2f} saat")

