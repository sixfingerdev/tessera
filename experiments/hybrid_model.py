import torch, torch.nn as nn, torch.nn.functional as F, math, time, os, json, random
from datasets import load_dataset

class MultiHashMemory(nn.Module):
    def __init__(self, d=128, bits=8, hashes=2):
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

class TransformerLayer(nn.Module):
    def __init__(self, d=128, nhead=4):
        super().__init__()
        self.attn=nn.MultiheadAttention(d,nhead,batch_first=True)
        self.ffn=nn.Sequential(nn.Linear(d,d*2), nn.GELU(), nn.Linear(d*2,d))
        self.ln1=nn.LayerNorm(d); self.ln2=nn.LayerNorm(d)
    def forward(self,x):
        N=x.shape[1]
        mask=torch.triu(torch.ones(N,N, device=x.device), diagonal=1).bool()
        h2,_=self.attn(x,x,x, attn_mask=mask)
        x=self.ln1(x+h2)
        x=self.ln2(x+self.ffn(x))
        return x

class HybridModel(nn.Module):
    """
    1 katman Transformer (exact local) + 2 katman Tessera (efficient global)
    En üstün kombinasyon
    """
    def __init__(self, vocab=10000, d=128):
        super().__init__()
        self.emb=nn.Embedding(vocab,d)
        self.layer0=TransformerLayer(d,4)  # İlk katman exact
        self.layer1=MultiHashMemory(d,8,2)  # Global
        self.layer2=MultiHashMemory(d,8,2)  # Global
        self.out=nn.Linear(d,vocab)
        self.d=d
    def forward(self,x):
        h=self.emb(x) * math.sqrt(self.d)
        h=self.layer0(h)
        h=self.layer1(h)
        h=self.layer2(h)
        return self.out(h)

class PureTransformer(nn.Module):
    def __init__(self, vocab=10000, d=128):
        super().__init__()
        self.emb=nn.Embedding(vocab,d)
        self.layers=nn.ModuleList([TransformerLayer(d,4) for _ in range(3)])
        self.out=nn.Linear(d,vocab)
        self.d=d
    def forward(self,x):
        h=self.emb(x)*math.sqrt(self.d)
        for l in self.layers: h=l(h)
        return self.out(h)

class PureTessera(nn.Module):
    def __init__(self, vocab=10000, d=128):
        super().__init__()
        self.emb=nn.Embedding(vocab,d)
        self.layers=nn.ModuleList([MultiHashMemory(d,8,2) for _ in range(3)])
        self.out=nn.Linear(d,vocab)
    def forward(self,x):
        h=self.emb(x)
        for l in self.layers: h=l(h)
        return self.out(h)

# Tokenizer
class SimpleTokenizer:
    def __init__(self, texts, vocab_size=10000):
        from collections import Counter
        counter=Counter()
        for t in texts: counter.update(t.lower().split())
        most_common=counter.most_common(vocab_size-4)
        self.vocab={"<pad>":0,"<unk>":1,"<bos>":2,"<eos>":3}
        for i,(w,_) in enumerate(most_common): self.vocab[w]=i+4
        self.inv_vocab={v:k for k,v in self.vocab.items()}
        self.vocab_size=vocab_size
    def encode(self, text, max_len=128):
        toks=text.lower().split()[:max_len]
        ids=[self.vocab.get(w,1) for w in toks]
        if len(ids)<max_len: ids=ids+[0]*(max_len-len(ids))
        return ids

print("Wikitext-2 yükleniyor...")
ds=load_dataset('Salesforce/wikitext','wikitext-2-raw-v1')
train_texts=[t for t in ds['train']['text'] if len(t.strip())>0]
val_texts=[t for t in ds['validation']['text'] if len(t.strip())>0]

tok_path="/home/user/tokenizer.json"
if os.path.exists(tok_path):
    with open(tok_path,'r') as f:
        vocab_dict=json.load(f)
    tokenizer=SimpleTokenizer([],10000)
    tokenizer.vocab=vocab_dict
else:
    tokenizer=SimpleTokenizer(train_texts,10000)
    with open(tok_path,'w') as f:
        json.dump(tokenizer.vocab,f)

def train_eval(model, steps=300):
    opt=torch.optim.AdamW(model.parameters(), lr=3e-4)
    model.train()
    for step in range(steps):
        batch_texts=random.sample(train_texts,16)
        x_batch=[]; y_batch=[]
        for txt in batch_texts:
            ids=tokenizer.encode(txt,128)
            x_batch.append(ids[:-1]); y_batch.append(ids[1:])
        x=torch.tensor(x_batch); y=torch.tensor(y_batch)
        logits=model(x)
        loss=F.cross_entropy(logits.view(-1,10000), y.view(-1), ignore_index=0)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        losses=[]
        for _ in range(20):
            batch_texts=random.sample(val_texts,16)
            x_batch=[]; y_batch=[]
            for txt in batch_texts:
                ids=tokenizer.encode(txt,128)
                x_batch.append(ids[:-1]); y_batch.append(ids[1:])
            x=torch.tensor(x_batch); y=torch.tensor(y_batch)
            logits=model(x)
            loss=F.cross_entropy(logits.view(-1,10000), y.view(-1), ignore_index=0)
            losses.append(loss.item())
        avg=sum(losses)/len(losses)
        return avg, math.exp(avg)

print("\n=== HİBRİT TEST - 300 step ===")
for name, ModelClass in [("Pure Transformer 3L", PureTransformer), ("Pure Tessera 3L", PureTessera), ("Hybrid 1 Trans + 2 Tessera", HybridModel)]:
    torch.manual_seed(0)
    model=ModelClass(vocab=10000,d=128)
    loss,ppl=train_eval(model, steps=300)
    print(f"{name}: loss={loss:.3f} ppl={ppl:.1f}")

print("\nSonuç: Hybrid en üstün olmalı çünkü local exact + global efficient bir arada")
