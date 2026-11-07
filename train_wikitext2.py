"""
Wikitext-2 ile Tessera vs Transformer - Checkpoint'li eğitim
120sn kill'e karşı her 100 step'te checkpoint kaydeder
"""
import torch, torch.nn as nn, torch.nn.functional as F
import os, math, time, json
from datasets import load_dataset

# --- Tokenizer basit ---
class SimpleTokenizer:
    def __init__(self, texts, vocab_size=10000):
        # Kelime frekansına göre vocab
        from collections import Counter
        counter = Counter()
        for t in texts:
            counter.update(t.lower().split())
        most_common = counter.most_common(vocab_size-4)
        self.vocab = {"<pad>":0, "<unk>":1, "<bos>":2, "<eos>":3}
        for i,(w,_) in enumerate(most_common):
            self.vocab[w]=i+4
        self.inv_vocab = {v:k for k,v in self.vocab.items()}
        self.vocab_size = vocab_size
    def encode(self, text, max_len=128):
        toks = text.lower().split()[:max_len]
        ids = [self.vocab.get(w,1) for w in toks]
        # pad
        if len(ids)<max_len:
            ids = ids + [0]*(max_len-len(ids))
        return ids
    def decode(self, ids):
        return " ".join([self.inv_vocab.get(i,"<unk>") for i in ids])

# --- Modeller ---
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

class TransformerModel(nn.Module):
    def __init__(self, vocab=10000, d=128, nhead=4, layers=2):
        super().__init__()
        self.emb=nn.Embedding(vocab,d)
        encoder_layer=nn.TransformerEncoderLayer(d_model=d, nhead=nhead, dim_feedforward=d*2, batch_first=True)
        self.encoder=nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.out=nn.Linear(d,vocab)
        self.d=d
    def forward(self,x):
        h=self.emb(x) * math.sqrt(self.d)
        # causal mask
        N=x.shape[1]
        mask=torch.triu(torch.ones(N,N, device=x.device), diagonal=1).bool()
        # TransformerEncoder expects float mask with -inf
        float_mask = torch.zeros(N,N, device=x.device).masked_fill(mask, float('-inf'))
        h=self.encoder(h, mask=float_mask)
        return self.out(h)

def train_one_model(model_name="tessera", steps_per_run=100, total_steps=1000):
    print(f"\n{'='*70}")
    print(f"EĞİTİM: {model_name.upper()} - Wikitext-2")
    print(f"{'='*70}")
    
    # Load dataset
    ds = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1')
    train_texts = [t for t in ds['train']['text'] if len(t.strip())>0]
    val_texts = [t for t in ds['validation']['text'] if len(t.strip())>0]
    print(f"Train: {len(train_texts)} docs, Val: {len(val_texts)} docs")
    
    # Tokenizer
    tok_path = "/home/user/tokenizer.json"
    if os.path.exists(tok_path):
        with open(tok_path,'r') as f:
            vocab_dict=json.load(f)
        tokenizer = SimpleTokenizer([], vocab_size=10000)
        tokenizer.vocab = vocab_dict
        tokenizer.inv_vocab = {v:k for k,v in vocab_dict.items()}
        tokenizer.vocab_size = 10000
        print("Tokenizer yüklendi")
    else:
        tokenizer = SimpleTokenizer(train_texts, vocab_size=10000)
        with open(tok_path,'w') as f:
            json.dump(tokenizer.vocab, f)
        print(f"Tokenizer oluşturuldu, vocab={len(tokenizer.vocab)}")
    
    # Model
    vocab_size=10000
    if model_name=="tessera":
        model = TesseraModel(vocab=vocab_size, d=128, layers=2, bits=8, hashes=2)
    else:
        model = TransformerModel(vocab=vocab_size, d=128, nhead=4, layers=2)
    
    device=torch.device('cpu')
    model.to(device)
    opt=torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    
    ckpt_dir=f"/home/user/checkpoints_{model_name}"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path=os.path.join(ckpt_dir, "ckpt.pt")
    log_path=os.path.join(ckpt_dir, "log.json")
    
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
        print(f"Checkpoint yüklendi, step {start_step}'den devam")
    else:
        print("Sıfırdan başlıyor")
    
    # Training loop - sadece steps_per_run kadar
    seq_len=128
    batch_size=16
    
    # Veriyi tokenize et ve batch'le
    # Basit: rastgele doc seç, encode et
    import random
    random.seed(42)
    
    model.train()
    step_times=[]
    for step in range(start_step, start_step+steps_per_run):
        if step>=total_steps:
            print(f"Hedef {total_steps} step'e ulaşıldı, duruyor")
            break
        t0=time.time()
        # Batch oluştur
        batch_texts = random.sample(train_texts, batch_size)
        x_batch=[]
        y_batch=[]
        for txt in batch_texts:
            ids=tokenizer.encode(txt, max_len=seq_len)
            x_batch.append(ids[:-1])
            y_batch.append(ids[1:])
        x=torch.tensor(x_batch, dtype=torch.long)
        y=torch.tensor(y_batch, dtype=torch.long)
        
        logits=model(x)
        loss=F.cross_entropy(logits.view(-1,vocab_size), y.view(-1), ignore_index=0)
        
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step()
        
        elapsed=time.time()-t0
        step_times.append(elapsed)
        ppl=math.exp(loss.item()) if loss.item()<10 else 9999
        
        if step%10==0:
            print(f"Step {step}: loss={loss.item():.3f} ppl={ppl:.1f} time={elapsed:.2f}s")
        logs.append({"step":step, "loss":loss.item(), "ppl":ppl, "time":elapsed})
        
        # Her 50 step'te checkpoint
        if step%50==0 and step>start_step:
            torch.save({"model":model.state_dict(), "opt":opt.state_dict(), "step":step+1}, ckpt_path)
            with open(log_path,'w') as f:
                json.dump(logs,f)
            print(f"Checkpoint kaydedildi step {step}")
    
    # Final checkpoint
    torch.save({"model":model.state_dict(), "opt":opt.state_dict(), "step":start_step+steps_per_run}, ckpt_path)
    with open(log_path,'w') as f:
        json.dump(logs,f)
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_losses=[]
        for _ in range(20):
            batch_texts = random.sample(val_texts, batch_size)
            x_batch=[]; y_batch=[]
            for txt in batch_texts:
                ids=tokenizer.encode(txt, max_len=seq_len)
                x_batch.append(ids[:-1]); y_batch.append(ids[1:])
            x=torch.tensor(x_batch); y=torch.tensor(y_batch)
            logits=model(x)
            loss=F.cross_entropy(logits.view(-1,vocab_size), y.view(-1), ignore_index=0)
            val_losses.append(loss.item())
        avg_val=sum(val_losses)/len(val_losses)
        print(f"\nValidation: loss={avg_val:.3f} ppl={math.exp(avg_val):.1f}")
    
    avg_time=sum(step_times)/len(step_times) if step_times else 0
    print(f"Ortalama step süresi: {avg_time:.3f}s, Tahmini 1 epoch (~{len(train_texts)//batch_size} step): {len(train_texts)//batch_size*avg_time/60:.1f} dakika")
    
    return logs

if __name__=="__main__":
    import sys
    name = sys.argv[1] if len(sys.argv)>1 else "tessera"
    train_one_model(name, steps_per_run=100, total_steps=1000)
