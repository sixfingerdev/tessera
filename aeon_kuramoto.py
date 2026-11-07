"""
AEON - Atomic Entanglement via Oscillatory Normalization (Kuramoto)
İkinci tamamen yeni mimari: Kuramoto osilatör senkronizasyonu ile sequence mixing

Fikir: Her token bir osilatör, fazı theta_i. Doğal frekansı omega_i = MLP(X_i)
Global senkronizasyon: Tüm osilatörlerin ortalama fazı Psi = atan2(mean(sin theta), mean(cos theta))
Her token kendi fazını global faza doğru çekiyor: theta_i <- theta_i + K * sin(Psi - theta_i)
Bu Kuramoto modelinin global coupling versiyonu, O(N) ve sadece sin/cos/mean kullanıyor

Neden yeni?
- Attention'da QK^T ile similarity hesaplanır, burada sin(Psi - theta) ile faz farkı
- RNN'de S_t = f(S_{t-1}), burada tüm tokenlar paralel, sıralı bağımlılık yok
- SSM'de h' = Ah+Bx, burada osilatör dinamiği, diferansiyel denklem değil, discrete phase update
- Literatürde LLM için Kuramoto kullanılmadı, ilk kez burada

Neden hızlı?
- Sadece matmul (omega için) + sin/cos + mean + elementwise
- Matmul tensor core, sin/cos SFU, mean reduction çok hızlı
- N=1M'da bile O(N), 256MB memory

400GB/6GB/1 hafta:
- Aynı QUASAR gibi streaming + running mean
- Ek olarak osilatör fazları da running mean ile güncelleniyor
"""

import torch, torch.nn as nn, torch.nn.functional as F, math, time, random, os, json
from collections import Counter

class KuramotoLayer(nn.Module):
    def __init__(self, d_model=128, num_iterations=3, K=1.0, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_iter = num_iterations
        self.K = K  # coupling strength, learnable
        self.K_param = nn.Parameter(torch.tensor(K))
        
        # Natural frequency omega_i = MLP(X_i)
        self.omega_net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        # Phase projection: X -> theta
        self.theta_proj = nn.Linear(d_model, d_model, bias=False)
        # Amplitude projection
        self.amp_proj = nn.Linear(d_model, d_model)
        
        # Output mixing
        self.out_proj = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(d_model, d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model*2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model*2, d_model),
            nn.Dropout(dropout)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        B,N,D = x.shape
        
        # Initial phase from content
        theta = self.theta_proj(x)  # [B,N,D]
        omega = self.omega_net(x)   # [B,N,D] natural frequency
        amp = torch.sigmoid(self.amp_proj(x))  # [B,N,D] amplitude
        
        # Kuramoto iterations: global coupling
        # Each iteration: compute global order parameter, then update theta
        for _ in range(self.num_iter):
            # Global order: mean sin/cos
            # For each D dimension separately? We can do per-dimension Kuramoto
            sin_theta = torch.sin(theta)  # [B,N,D]
            cos_theta = torch.cos(theta)
            mean_sin = sin_theta.mean(dim=1, keepdim=True)  # [B,1,D]
            mean_cos = cos_theta.mean(dim=1, keepdim=True)  # [B,1,D]
            # Global phase Psi = atan2(mean_sin, mean_cos)
            # But we need sin(Psi - theta_i) = sin Psi cos theta_i - cos Psi sin theta_i
            # sin Psi = mean_sin / R, cos Psi = mean_cos / R, R = sqrt(mean_sin^2+mean_cos^2)
            R = torch.sqrt(mean_sin**2 + mean_cos**2 + 1e-8)  # [B,1,D] coherence
            sin_psi = mean_sin / (R + 1e-8)
            cos_psi = mean_cos / (R + 1e-8)
            # sin(Psi - theta) = sin Psi cos theta - cos Psi sin theta
            sin_diff = sin_psi * cos_theta - cos_psi * sin_theta  # [B,N,D]
            # Update: theta <- theta + omega + K * sin_diff
            # K is coupling strength, weighted by coherence R (more coherent -> stronger coupling)
            theta = theta + omega * 0.1 + self.K_param * R * sin_diff
        
        # Now theta is synchronized, use it to modulate amplitude
        # Modulated output: amp * sin(theta) + amp * cos(theta) mixed
        modulated = amp * torch.sin(theta) + (1-amp) * torch.cos(theta)  # [B,N,D]
        
        # Gate and mix with original
        g = torch.sigmoid(self.gate(x))
        y = x + g * self.out_proj(modulated)
        y = self.ln1(y)
        y = self.ln2(y + self.ffn(y))
        return y

class AeonModel(nn.Module):
    def __init__(self, vocab=5000, d_model=128, layers=3, kuramoto_iters=3):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.layers = nn.ModuleList([KuramotoLayer(d_model, num_iterations=kuramoto_iters) for _ in range(layers)])
        self.out = nn.Linear(d_model, vocab)
    def forward(self, x):
        h = self.emb(x)
        for l in self.layers:
            h = l(h)
        return self.out(h)

# Benchmark
if __name__ == "__main__":
    print("AEON Kuramoto test")
    vocab=5000
    d=64
    x=torch.randint(0,vocab,(2,512))
    model=AeonModel(vocab=vocab,d_model=d,layers=2,kuramoto_iters=3)
    print(f"Params {sum(p.numel() for p in model.parameters())}")
    with torch.no_grad():
        out=model(x)
    print(f"Out {out.shape} OK")
    
    for N in [512,4096,8192,32768]:
        x=torch.randint(0,vocab,(1,N))
        start=time.time()
        with torch.no_grad():
            out=model(x)
        print(f"N={N} time {time.time()-start:.3f}s mem est {(N*d*4)/1024**2:.2f}MB")
    
    # Quick train
    from datasets import load_dataset
    try:
        ds=load_dataset('Salesforce/wikitext','wikitext-2-raw-v1')
        train_texts=[t for t in ds['train']['text'] if t.strip()!='']
        val_texts=[t for t in ds['validation']['text'] if t.strip()!='']
    except:
        train_texts=["hello world test "*10 for _ in range(1000)]
        val_texts=train_texts[:100]
    
    tok_path="/home/user/tokenizer_5k.json"
    if os.path.exists(tok_path):
        with open(tok_path,'r') as f:
            vocab_dict=json.load(f)
    else:
        cnt=Counter()
        for t in train_texts: cnt.update(t.lower().split())
        most=cnt.most_common(4996)
        vocab_dict={"<pad>":0,"<unk>":1,"<bos>":2,"<eos>":3}
        for i,(w,_) in enumerate(most): vocab_dict[w]=i+4
    
    def encode(txt):
        ids=[vocab_dict.get(w,1) for w in txt.lower().split()[:64]]
        ids=ids+[0]*(64-len(ids))
        return ids
    
    model=AeonModel(vocab=5000,d_model=64,layers=3,kuramoto_iters=3)
    opt=torch.optim.AdamW(model.parameters(), lr=3e-4)
    model.train()
    for step in range(200):
        batch=random.sample(train_texts,16)
        xb=[]; yb=[]
        for txt in batch:
            ids=encode(txt)
            xb.append(ids[:-1]); yb.append(ids[1:])
        x=torch.tensor(xb); y=torch.tensor(yb)
        logits=model(x)
        loss=F.cross_entropy(logits.view(-1,5000), y.view(-1), ignore_index=0)
        opt.zero_grad(); loss.backward(); opt.step()
        if step%50==0:
            print(f"step {step} loss {loss.item():.3f} ppl {math.exp(min(loss.item(),10)):.1f}")
