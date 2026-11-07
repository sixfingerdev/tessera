# Derinlemesine Araştırma Notları - Transformers Ekosistemi

**Tarih:** 2026-09-28
**Konular:** Transformers GitHub Reposu, AdamW, GaLore, Linear Attention, DeepSeek v4.1 Flash

---

## 1. TRANSFORMERS MİMARİSİ - GitHub Reposu ve "Attention Is All You Need"

### 1.1 GitHub Reposu Yapısı (huggingface/transformers)
- **167k stars, 34.7k forks, 24k+ commit** - En büyük açık kaynak ML reposu
- Ana dizin: `src/transformers/models/` altında 200+ model ailesi
  - Her model için: `configuration_*.py`, `modeling_*.py`, `tokenization_*.py`
- Örnek: `bert/modeling_bert.py` (1370 satır)
  - `BertEmbeddings`: word + position + token_type embeddings, LayerNorm, dropout
  - `BertSelfAttention`, `BertAttention`, `BertLayer`, `BertEncoder` katmanları
  - `ALL_ATTENTION_FUNCTIONS` ile FlashAttention, SDPA, eager attention desteği
  - `Cache`, `DynamicCache`, `EncoderDecoderCache` - KV cache yönetimi
  - `GradientCheckpointingLayer` - bellek optimizasyonu
- Modern mimarilerde (LLaMA, Mistral): `RMSNorm` (LayerNorm yerine), `SwiGLU` aktivasyon, `RoPE` positional encoding
- `modeling_layers` ve `modular_transformers` yapısı - son versiyonda modüler kod üretimi

### 1.2 Attention Is All You Need (Vaswani et al. 2017) - Derin Okuma
**Orijinal Paper: arxiv.org/html/1706.03762v7**

**Temel Fikir:** RNN/CNN tamamen kaldırıldı, sadece attention kullanıldı. Sequential nature precludes parallelization - bu yüzden self-attention.

**Encoder:**
- N=6 katman, her katman: Multi-Head Self-Attention + Position-wise FFN
- Residual: `LayerNorm(x + Sublayer(x))`, d_model=512
- Tüm sub-layer'lar 512 boyutunda çıktı veriyor residual için

**Decoder:**
- N=6 katman, 3 sub-layer: Masked Self-Attention + Encoder-Decoder Attention + FFN
- Masked attention: illegal connections softmax öncesi -inf ile maskeleniyor, autoregressive property korunuyor
- Encoder-decoder attention: Q önceki decoder katmanından, K,V encoder çıktısından geliyor

**Scaled Dot-Product Attention:**
```
Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V
```
- d_k büyük olduğunda dot product büyüyor, softmax gradyanı küçülüyor -> sqrt(d_k) ile scale
- Q,K,V matrisler halinde paketlenip paralel hesaplanıyor

**Multi-Head Attention:**
- h=8 head, her biri d_k=d_v=d_model/h=64
- Her head için ayrı W^Q, W^K, W^V projeksiyonları, sonra concat + W^O
- Amaç: Farklı representation subspace'lerden bilgiye ortak attend etmek. Tek head averaging ile bunu engelliyor.
- Maliyet tek head ile benzer çünkü boyut küçülüyor

**Position-wise FFN:**
```
FFN(x) = max(0, xW1 + b1)W2 + b2
```
- Kernel size 1 olan iki conv gibi, her pozisyona aynı ama katmanlar arası farklı parametreler
- d_model=512, d_ff=2048

**Positional Encoding:**
- Sinusoidal: PE(pos,2i)=sin(pos/10000^(2i/d_model)), PE(pos,2i+1)=cos(...)
- Wavelength 2π'den 10000*2π'ye geometric progression
- Sebep: Herhangi fixed offset k için PE_{pos+k} linear function of PE_pos olarak temsil edilebilir -> relative position öğrenmesi kolay
- Learned positional embedding ile neredeyse aynı sonuç, ama sinusoidal extrapolate edebiliyor daha uzun sequence'lere

**Why Self-Attention? - Karşılaştırma Tablosu:**
- Self-Attention: O(n²·d) complexity, O(1) sequential ops, O(1) max path length
- Recurrent: O(n·d²), O(n) sequential, O(n) path
- Convolutional: O(k·n·d²), O(1) sequential, O(log_k(n)) path
- Restricted Self-Attention: O(r·n·d), O(1), O(n/r)
- Sonuç: n < d olduğunda self-attention recurrent'ten hızlı, path length constant olduğu için long-range dependency öğrenmesi çok daha kolay

**Modern Evrim:**
- Decoder-only modeller (GPT, LLaMA, Chinchilla) sadece masked self-attention + FFN kullanıyor, cross-attention yok
- Pre-norm vs post-norm: Orijinal post-norm (LayerNorm sonrası), modern modeller pre-norm kullanıyor stabilite için
- Aktivasyon: ReLU -> GELU -> SwiGLU

---

## 2. ADAMW OPTIMIZER - Decoupled Weight Decay Regularization

### 2.1 Paper: arxiv.org/html/1711.05101v3 (Loshchilov & Hutter, ICLR 2019)

**Problem Tespiti:**
- L2 regularization ve weight decay SGD için eşdeğer (learning rate ile rescale edildiğinde)
- Ama adaptive gradient algoritmaları (Adam, RMSProp) için EŞDEĞER DEĞİL - çoğu kütüphane bunu gözden kaçırıyor ve L2'yi weight decay diye yanlış isimlendiriyor
- Bu yüzden Adam, SGD with momentum'dan daha kötü generalize ediyor image classification'da

**SGD için Eşdeğerlik (Proposition 1):**
Standard SGD weight decay λ ile:
```
θ_{t+1} = (1-λ)θ_t - α∇f_t(θ_t)
```
L2 regularization λ' ile:
```
f_t^{reg}(θ) = f_t(θ) + λ'/2 ||θ||², λ' = λ/α
```
Aynı adımları yürütüyor. Ama λ' learning rate α'ya tightly coupled - optimal λ' bulmak için α ile birlikte aramak gerekiyor.

**Adam için Eşdeğersizlik (Proposition 2):**
Adaptive optimizer: θ_{t+1} ← θ_t - α M_t ∇f_t(θ_t), M_t ≠ kI (preconditioner)
Weight decay ile: θ_{t+1} ← (1-λ)θ_t - α M_t ∇f_t(θ_t)
L2 regularization ekleyerek aynı davranışı verecek λ' YOK. Çünkü L2 gradyanı da M_t ile scale ediliyor.

**Intuition:**
- L2 regularization: loss gradyanı + regularizer gradyanı toplanıp birlikte adapt ediliyor
- Büyük historik gradyanı olan weight'ler daha az regularize ediliyor (çünkü gradyanı normalize ediliyor)
- Decoupled weight decay: Sadece loss gradyanı adapt ediliyor, weight decay ayrı adımda tüm weight'lere aynı rate λ ile uygulanıyor
- Sonuç: Büyük s_i olan parametreler L2'ye göre daha fazla regularize ediliyor

**Scale-adjusted L2 (Proposition 3 - Fixed preconditioner):**
M_t = diag(s)^-1 sabit ise:
```
f_t^{sreg}(θ) = f_t(θ) + λ'/(2α) ||θ ⊙ sqrt(s)||²
```
Yani L2'nin eşdeğeri s ile scale edilmiş bir norm oluyor.

**Bayesian Filtering Justification (Aitchison 2018):**
- Stochastic optimization Bayesian filtering olarak görülüyor: optimal parametrelerin dağılımını infer etmek
- State transition prior: P(θ_{t+1}|θ_t) = N((I-A)θ_t, Q), A=λI
- Bu tam decoupled weight decay: her adımda mean'i (1-λ) ile çarpmak
- L2 regularization bu framework'te doğal çıkmıyor, weight decay çıkıyor

**AdamW Algoritması (Pseudocode):**
```
g_t = ∇f_t(θ_{t-1}) + λθ_{t-1}  # L2 için, AdamW için sadece ∇f_t
m_t = β1 m_{t-1} + (1-β1) g_t
v_t = β2 v_{t-1} + (1-β2) g_t²
m̂_t = m_t / (1-β1^t)
v̂_t = v_t / (1-β2^t)
η_t = SetScheduleMultiplier(t)  # cosine annealing vb
θ_t = θ_{t-1} - η_t (α m̂_t / (sqrt(v̂_t)+ε) + λ θ_{t-1})  # decoupled!
```
Yeşil kısım: weight decay adaptive update'ten sonra ayrı uygulanıyor

**Empirik Bulgular:**
- Decoupled weight decay optimal learning rate ve weight decay seçimini birbirinden ayırıyor (heatmap diagonal olmaktan çıkıyor)
- AdamW CIFAR-10 ve ImageNet32x32'de %15 relative improvement test error'da
- Scheduled learning rate multiplier (cosine annealing) Adam için bile çok faydalı, adaptive olmasına rağmen
- Optimal weight decay toplam batch pass sayısı ile ters orantılı: daha uzun eğitim -> daha küçük optimal decay

**Pratik Kullanım:**
- PyTorch: torch.optim.AdamW, parameter grupları ile bias ve LayerNorm weight'lerine decay uygulamamak yaygın
- Genelde kullanılan: β1=0.9, β2=0.999, ε=1e-8, λ=0.01-0.1, lr warmup

---

## 3. GALORE OPTIMIZER - Gradient Low-Rank Projection

### 3.1 Paper: arxiv.org/html/2403.03507v2 (Zhao et al., ICML 2024 Oral)

**Motivasyon:**
- LLM eğitimi bellek canavarı: LLaMA 7B tek batch için 58GB (14GB params + 42GB Adam states + gradients + 2GB activations) BF16'da
- LoRA: W = W0 + BA, r << min(m,n), sadece A,B trainable, optimizer states küçük ama:
  - Full-rank training'den düşük performans
  - Training dynamics değişiyor (reparametrization)
  - Pre-training için full-rank warmup gerekiyor (ReLoRA)

**GaLore Fikri:**
- Weight matrix low-rank olmayabilir, ama GRADIENT low-rank oluyor training sırasında!
- Teorik kanıt: Reversible networks için gradient formu

**Teori - Gradient Formu (Theorem 3.2):**
Chained reversible network için layer l gradient:
```
G_l = (J_l^T y - J_l^T J_l W_l f_{l-1}) f_{l-1}^T   # L2 loss için
```
Batch size N için:
```
G = 1/N Σ (A_i - B_i W C_i), B_i, C_i PSD
```

**Lemma 3.3 - Gradient Low-Rank Oluyor:**
Vanilla SGD: W_t = W_{t-1} + η G_{t-1}
S = 1/N Σ C_i ⊗ B_i, λ1<λ2 en küçük iki eigenvalue
```
sr(G_t) ≤ sr(G_{t0}^{∥}) + ((1-ηλ2)/(1-ηλ1))^{2(t-t0)} * ||G0 - G_{t0}^{∥}||_F² / ||G_{t0}^{∥}||_2²
```
İkinci terim exponential decay -> sr(G_t) düşük kalıyor
Corollary 3.4: Eğer G_t = 1/N Σ (a_i - B_i W f_i) f_i^T ve B_i full-rank, rank({f_i}) = N' < n ise sr(G_{t0}^{∥}) ≤ n-N' ve büyük t için sr(G_t) ≤ n/2
Corollary 3.5: V1 1-dim ve decomposable ise sr=1, rank-1 gradient!

Transformers için: FFN lower layer (project-up) weight gradienti JoMA framework ile low-rank olduğu ispatlanıyor

**GaLore Tanımı (Definition 3.6):**
```
W_T = W0 + η Σ_{t=0}^{T-1} G̃_t
G̃_t = P_t ρ_t(P_t^T G_t Q_t) Q_t^T
```
P_t ∈ R^{m×r}, Q_t ∈ R^{n×r} projection matrices, ρ_t Adam gibi entry-wise regularizer
LoRA'dan farkı: Explicit low-rank updates, ek adaptor yok, training dynamics değişmiyor. r=min(m,n) olduğunda ρ_t=1 ile orijinal model ile aynı trajectory!

**Convergence (Theorem 3.8):**
Sabit P,Q ile, gradient formu Eq 8 ise ve A_i, B_i, C_i LL-continuous, ||W_t||≤D, R_t:=P^T G_t Q, B̂_it:=P^T B_i P, Ĉ_it:=Q^T C_i Q, κ_t:=1/N Σ λ_min(B̂_it)λ_min(Ĉ_it)
```
||R_t||_F ≤ [1-η(κ_{t-1}-L_A-L_B L_C D²)] ||R_{t-1}||_F
```
Eğer min κ_t > L_A + L_B L_C D² ise R_t→0, GaLore converge ediyor. P,Q B̂ ve Ĉ'nin en büyük eigenvector subspace'lerine projekte etmeli ki κ büyük olsun -> SVD ile G_t spektrumunu kullanmak mantıklı.

**SVD ile Projeksiyon (Eq 12-13):**
```
G_t = U S V^T ≈ Σ_{i=1}^r s_i u_i v_i^T
P_t = [u1,...,ur], Q_t = [v1,...,vr]
```
Her T adımda (örn 200) recompute ediliyor.

**Memory-Efficient Optimization (Algorithm 2 - Adam with GaLore):**
```
Input: W∈R^{m×n}, m≤n, rank r, freq T
M0∈R^{n×r}=0, V0∈R^{n×r}=0
for t:
  G_t = -∇_W φ_t(W_t)
  if t mod T == 0:
    U,S,V = SVD(G_t)
    P_t = U[:,:r]
  else P_t = P_{t-1}
  R_t = P_t^T G_t  # compact space
  M_t = β1 M_{t-1} + (1-β1) R_t
  V_t = β2 V_{t-1} + (1-β2) R_t²
  N_t = M_t / (sqrt(V_t)+ε)
  G̃_t = α P N_t  # back to original
  W_t = W_{t-1} + η G̃_t
```
Sadece bir projeksiyon matrix kullanılıyor (m≤n ise P, değilse Q) en iyi trade-off için.

**Memory Hesabı:**
- W: mn
- GaLore: mn + mr + 2nr (proj matrix + 2 low-rank moments)
- LoRA: mn + 3mr + 3nr (W0 + BA + optimizer states for A,B)
- Örn m=n=4096, r=512: GaLore %65.5 optimizer memory azaltıyor, 8-bit GaLore %82.5!

**Combining Techniques:**
- 8-bit Adam: GaLore doğrudan 8-bit Adam üzerine uygulanıyor
- Per-layer weight updates: Gradient'i tüm layer'lar için tutmak yerine backprop sırasında update yapma (LOMO tekniği) - fused backward

**Hyperparametreler:**
- rank r (LoRA'da da var)
- subspace change frequency T: 50-1000 arası fark yok, overhead <10%, sweet spot var
- scale factor α: LoRA'daki α/r gibi ama r'ye bağlı değil, pre-training'de küçük r için α/r convergence'ı çok etkiliyor

**Deneyler:**
- C4 dataset, LLaMA architecture (RMSNorm, SwiGLU), BF16
- 60M,130M,350M,1B modeller: GaLore full-rank'a yakın perplexity (örn 1B: full 15.56, GaLore 15.64) ama %50 memory az, LoRA 19.21, ReLoRA 18.33 çok daha kötü
- LLaMA 7B 150K step (19.7B token): 8-bit GaLore 14.65 ppl, 8-bit Adam 14.61, memory 18G vs 26G
- GLUE fine-tuning RoBERTa-Base rank 4: GaLore 85.89 avg, LoRA 85.61
- İlk kez 7B modeli tek RTX 4090 24GB'de from scratch train etmek mümkün!

**GaLore 2 ve GUM:**
- GaLore 2: Daha büyük scale, low-bit quantization, higher-order tensor
- GUM (GaLore Unbiased with Muon): Muon optimizer ile unbiased low-rank, layerwise sampling ile bias düzeltme

---

## 4. LINEAR ATTENTION MEKANİZMASI

### 4.1 Orijinal Paper: arxiv.org/html/2006.16236v3 - Transformers are RNNs

**Problem:**
Softmax attention O(N²) time ve memory, N sequence length. Çok uzun sequence'ler için prohibitively slow.

**Genelleştirilmiş Attention (Eq 3):**
```
V'_i = Σ_j sim(Q_i,K_j) V_j / Σ_j sim(Q_i,K_j)
```
Softmax için sim(q,k)=exp(q^T k / sqrt(D))

**Kernel Trick:**
sim non-negative olmalı, kernel k(x,y) olarak görülebilir. Feature map φ(x) ile:
```
sim(q,k) = φ(q)^T φ(k)
```
O zaman:
```
V'_i = φ(Q_i)^T Σ_j φ(K_j) V_j^T / φ(Q_i)^T Σ_j φ(K_j)   (Eq 5)
```
Vectorized:
```
(φ(Q) φ(K)^T) V = φ(Q) (φ(K)^T V)   (Eq 6)
```
Sol taraf O(N²), sağ taraf O(N)! Çünkü Σ_j φ(K_j) V_j^T ve Σ_j φ(K_j) tüm query'ler için bir kez hesaplanıp reuse ediliyor.

**Complexity:**
- Softmax: O(N² max(D,M))
- Linear: Önce φ map C boyutuna, sonra O(N C M)
- Exponential kernel feature map sonsuz boyutlu -> linearization infeasible
- Polynomial kernel degree 2: O(N D² M), N>D² olduğunda avantajlı (tens of thousands sequence)
- Pratikte: φ(x)=elu(x)+1 (Eq 7), ReLU yerine ELU gradyan 0 olmasın diye. O(N D M) maliyet, softmax ile aynı performans ama çok daha az memory.

**Causal Masking (Eq 8-9):**
Autoregressive için i-th position sadece j≤i'ye attend edebilmeli:
```
V'_i = Σ_{j=1}^i sim(Q_i,K_j) V_j / Σ_{j=1}^i sim(Q_i,K_j)
     = φ(Q_i)^T Σ_{j=1}^i φ(K_j) V_j^T / φ(Q_i)^T Σ_{j=1}^i φ(K_j)
```
S_i = Σ_{j=1}^i φ(K_j) V_j^T, Z_i = Σ_{j=1}^i φ(K_j) tanımla:
```
V'_i = φ(Q_i)^T S_i / φ(Q_i)^T Z_i   (Eq 12)
```
S_i, Z_i S_{i-1}, Z_{i-1}'den constant time'da hesaplanıyor -> linear complexity causal için de!

**Gradient Computation (Eq 13-15):**
Naive implementation tüm S_i'leri saklamalı -> memory max(D,M) kat artıyor. Cumulative sum olarak gradyan türetiliyor:
```
∇_{φ(Q_i)} L = ∇_{V̄_i} L (Σ_{j=1}^i φ(K_j) V_j^T)^T
∇_{φ(K_i)} L = (Σ_{j=i}^N φ(Q_j) (∇_{V̄_j} L)^T) V_i
∇_{V_i} L = (Σ_{j=i}^N φ(Q_j) (∇_{V̄_j} L)^T)^T φ(K_i)
```
Cumulative sum linear time, constant memory.

**Training ve Inference:**
- Training: Full ground truth var, layerwise paralelleştirilebilir, GPU avantajı
- Inference: Output timestep i, input timestep i+1 -> paralelleştirilemez, transformer'da cost O(N²) çünkü tüm önceki timesteplere attention
- Linear transformer: Best of both worlds! Training paralelleştirilebilir, inference'de φ(K_j)V_j^T matrix'ini internal state olarak saklayıp her adımda update ediyor RNN gibi -> constant time ve memory per prediction, 4000x hızlanma çok uzun sequence'lerde!

**Transformers are RNNs (Eq 16-20):**
Causal masking'li herhangi transformer layer RNN olarak yazılabilir:
```
s0=0, z0=0
s_i = s_{i-1} + φ(x_i W_K)(x_i W_V)^T
z_i = z_{i-1} + φ(x_i W_K)
y_i = f_l( φ(x_i W_Q)^T s_i / φ(x_i W_Q)^T z_i + x_i )
```
İki hidden state: attention memory s ve normalizer memory z. Depth değil time'a göre recurrence (Universal Transformer'dan farkı).

**Feature Map Seçimleri ve Gelişmeler:**
- ELU+1, ReLU, exp gibi basit map'ler
- Performer: FAVOR+ random features ile softmax kernel unbiased approximate
- DPFP: Deterministic parameter-free projection, approximate orthogonality ile memory capacity artırma
- Taylor expansion: exponential kernel'i higher-order approximate
- Gated Linear Attention (GLA): Data-dependent gating, S_t = γ S_{t-1} + k_t^T v_t, γ input-conditioned decay (RetNet, Mamba, xLSTM, GateLoop benzeri)
- FlashLinearAttention: Hardware-efficient, chunkwise, tiling, recomputation-materialization trade-off, FlashAttention-2'den bile hızlı short sequence'lerde (1K)
- TransNormer: Scaling yerine normalization ile gradient stabilizasyonu + diagonal attention ile local context

**Hibrit Modeller - En Güvenilir Yol:**
Pure linear attention complex reasoning ve ultra-long retrieval'de performans kaybı yaşıyor. Consensus: hybrid en iyi
- Jamba: 8 Mamba layer + 1 Transformer layer
- MiniMax-01: Çoğu layer lightning attention, her 8 layer'da bir softmax full attention
- Bu modeller standard transformer'ı koruyor hatta geçiyor, ama near-lossless performans ile inference hızı sağlıyor

---

## 5. DEEPSEEK V4.1 FLASH - Son Nesil Mimari

### 5.1 Genel Bakış
- **Release:** 10 Eylül 2026, en küçük model yeni ailede ama ground-up redesign
- **Toplam Parametre:** 748B = 552B backbone + 196B Engram conditional memory
- **Aktif Parametre:** 8B prefill, 16B decode - asymmetric!
- **Context:** 1M token (1,048,576), max output 384K token
- **Mimari:** Causal Encoder-Decoder (CED) 40 layer (20 encoder + 20 decoder)
- **Attention:** CSA2 (Compressed Sparse Attention 2) + FP4 KV cache + SWA Bounded Replay
- **MoE:** 384 routed + 1 shared expert, top-6 aktif, SwiGLU
- **Multimodal:** Native vision, DeepSeek-ViT with 2D-RoPE, 672x672 -> 3x3 pixel unshuffle -> 224x224 grid, 16x16 patch, 196 token per image
- **Diğer:** Single-Pass mHC residual, Engram memory, DSpark speculative decoding
- **Lisans:** MIT, HuggingFace'de açık ağırlıklar

### 5.2 Neden Önemli? Efficiency Inversion
V4 Pro: 1.6T total, 49B aktif, GPQA 92.4, DeepSWE 62.7, Codeforces 3348
V4.1 Flash: 552B total, 8B/16B aktif, GPQA 90.9, DeepSWE 74.2 (+11.5!), Codeforces 3471, HLE with tools 63.9 vs 60.0, Terminal-Bench 90.6 vs 82.7
**Flash tier Pro'yu her benchmark'te geçiyor, 1/3 total param, 1/6 aktif param ile!** İlk kez flash pro'yu tamamen replace ediyor. 14 Eylül 2026'dan itibaren deepseek-v4-pro istekleri v4.1-flash'e route ediliyor.

### 5.3 Causal Encoder-Decoder (CED) - YOCO'dan İlham
**Yapı:** 20 layer causal encoder + 20 layer causal decoder, her ikisi de causal (bidirectional değil)

**Prefill:**
1. Input tokenlar 20 encoder layer'dan geçiyor, her layer CSA2 + MoE FFN, çıktı H_enc [seq_len, d_model]
2. Final encoder hidden state H_enc[-1] linear projeksiyon ile **global KV cache** oluşturuyor [seq_len, num_heads, head_dim]
3. Decoder layer'lar bu global KV cache'e attend ediyor, kendi layer'larının KV'sine değil

**Decode:**
1. Yeni token 20 encoder layer'dan geçiyor -> H_enc_new
2. Final hidden state projeksiyonu global KV cache'e append ediliyor
3. Decoder'lar büyüyen global KV cache'e attend ediyor

**Neden Önemli:**
Traditional decoder-only 40 layer modelde 40 ayrı KV cache var. V4.1'de decoder KV cache'i 20 decoder layer arasında paylaşılıyor (hepsi aynı projected global KV'ye attend ediyor) -> 20x azaltma! Encoder per-layer KV'leri prefill sonrası discard ediliyor, sadece final projected global KV kalıyor. Decode'da encoder incremental KV'leri de projeksiyon sonrası hemen discard. Sonuç: **Tüm decode fazı boyunca sadece bir KV cache tensor persist ediyor!**

**Doğrulama:** Model card "890 bytes per token" vs V4 Flash 3456 bytes -> 4x compression, CED ile tutarlı.

### 5.4 Compressed Sparse Attention 2 (CSA2)
**Üç statik mod, training'de sabit, input'a göre değişmiyor:**

1. **Full Attention:** Tüm tokenlara attend, global context kritik yerlerde (early encoder, late decoder)
2. **Reindex Attention:** Top-K en relevant token seç (K tipik 1024-2048), Hierarchical Sparse Indexer ile relevance score hesapla, O(n²) -> O(K·n)
3. **Reuse Attention:** En yakın önceki Reindex layer'ın seçtiği indexleri reuse et, indexer maliyetini ortadan kaldır

**Hierarchical Sparse Indexer (Reindex için):**
- Stage 1 Coarse: Sequence'ü chunk'lara böl (16 token/chunk), chunk-level relevance, Top-M chunk seç
- Stage 2 Fine: Seçilen chunk'larda per-token relevance, Top-K token seç
- Maliyet O(n log n) uzun sequence'lerde

**Cross-Layer KV Sharing:**
Birden fazla ardışık layer aynı KV cache tensor'ını paylaşabiliyor. Örn Layer 10-13 hepsi Reuse ve Layer 9 indexlerini referanslıyorsa, Layer 9'un cached K,V projeksiyonlarını recompute etmeden reuse ediyorlar. Model card "effective unique KV caches 40'tan ~12'ye düşüyor" diyor, CSA1'den CSA2'ye 3.5x compression.

**FP4 KV Cache Compression (E2M1):**
- Format: 1 sign, 2 exponent, 1 mantissa bit (4-bit float)
- Lossy ama minimal quality loss: GPQA Diamond 91.1 (FP8) -> 90.9 (FP4), -0.2 point
- BF16'a göre 4x, FP8'e göre 2x azaltma
- CED (20x) + CSA2 sharing (3.5x) + FP4 (4x) kombinasyonu kümülatif ~4x KV cache reduction vs V4 Flash (890 vs 3456 bytes/token)

**Neden Önemli:**
Long-context inference (1M token) KV-cache-bound, compute-bound değil. 1M token * 3456 bytes = 3.2GB VRAM per request, 890 bytes ile 850MB -> aynı hardware'de 3.7x throughput!

### 5.5 SWA Bounded Replay
**SWA = Sliding Window Attention**
Traditional: Son N token'ın (örn 32K) KV cache'ini sakla, token T+1 generate ederken [T-N+1,T] aralığına attend et. 40 layer için 40×N×sizeof(KV) byte.

Bounded Replay:
1. Sadece son n_win token'ın (tipik 4096-8192) input token ID'lerini sakla (2 byte per ID vs 88 byte per token FP4 KV)
2. Yeni token generate ederken o n_win token için forward pass replay yap, KV state'leri on-the-fly reconstruct et
3. Reconstructed KV ile attention yap, sonra discard et

"Bounded" n_win ile sınırlı replay demek, full window (32K) değil. Küçük recompute maliyeti karşılığında büyük memory saving. Persistent KV cache'i ~8x azaltıyor (88 -> 11 byte/token). Toplam 4x reduction'a katkıda bulunuyor.

### 5.6 Single-Pass mHC Residual Connections
**mHC = Mega-History-Compression**
Standard residual: output = input + layer(input)
mHC: Tüm önceki layer output'larının compressed representation'ı ile learned linear mixing

Her layer L'de:
1. layer_output = Attention(input) + FFN(input)
2. Running history compression vector H = H + compress(layer_output), compress = learned linear proj (d_model -> d_model/8)
3. output = W1 * layer_output + W2 * H, W1,W2 learned scalar weights

"Single-Pass" incremental tek forward pass'te history compression hesaplamak demek.

Neden: Very deep modellerde (>40 layer) vanishing gradient, mHC her layer'ın tüm önceki layer'ların compressed summary'sine direkt erişmesini sağlıyor, training stability ve convergence speed artırıyor. Changelog: "mHC improves training stability and convergence speed for models with >40 layers"

### 5.7 MoE Configuration
- Total experts: 384 routed + 1 shared
- Aktif: 6 routed + 1 shared = 7 total per token
- Expert size: 2.1B param per expert (FFN)
- Routing: Token-level Top-K (K=6), gating network: scores = softmax(W_gate · token_embedding)
- FFN: `FFN_output = sum(score_i · Expert_i(token) for i in top-6) + Expert_shared(token)`, shared her zaman aktif, baseline transformation
- SwiGLU: `FFN(x) = (Swish(W1·x) ⊙ (W2·x)) · W3`, Swish(x)=x·sigmoid(x), GLaM, PaLM'de ReLU/GELU'dan daha iyi
- Load balancing loss: `L_balance = α · sum((f_i - 1/N)²)`, f_i expert i'ye route edilen token fraction, N=384, α~0.01, expert collapse önlemek için uniform usage teşvik

### 5.8 Engram Conditional Memory - 196B
- Ana Transformer backbone'un parçası değil, ayrı lookup table
- Sadece decode'da aktif, prefill'de değil -> 8B vs 16B aktif farkını açıklıyor
- Her yeni token ID t için: `engram_embedding = Engram_table[hash(t)]`
- Decoder hidden state'e concat/add edilip MoE FFN öncesi
- Sadece ~1-2% Engram parametresi per token access -> ~2B ek aktif parametre decode'da
- Amaç: Context window'dan bağımsız long-term memory, rare entity için factual association saklama (örn "François-Marie Arouet" -> "Voltaire real name"), RAG ihtiyacını azaltma, factual recall iyileştirme

### 5.9 DSpark Speculative Decoding
- Semi-autoregressive: Küçük draft model (tahmini ~1B, ayrı train edilmiş) K candidate token (4-8) paralel generate ediyor, prompt + önceden kabul edilen tokenlara conditioned
- Ana V4.1 Flash tek forward pass'te tüm K candidate için acceptance score hesaplıyor
- Score > threshold (0.9) ise kabul, reject varsa o pozisyondan itibaren V4.1 Flash ile regenerate
- Confidence-scheduled verification: Acceptance threshold draft model'in current sequence için historical accuracy'sine göre adaptif
- Changelog: "1.6x speedup average multi-turn conversations, up to 2.3x short responses (<50 tokens)"

### 5.10 Vision Architecture
- Vision encoder: DeepSeek-ViT custom, from scratch train edilmiş, 2D-RoPE
- Input 672x672, 3-way pixel unshuffle -> 224x224 grid, patch 16x16 -> 14x14 grid = 196 vision token per image
- 2D-RoPE: Standard RoPE 1D scalar position, vision için (row,col) pair, rotation angles her iki dimension'dan: `[cosθ_row·patch, sinθ_row·patch, cosθ_col·patch, sinθ_col·patch]`, spatial structure koruyor, relative position reasoning (top-left corner gibi)
- Cross-attention: Vision tokenlar text'e, text vision'a attend ediyor
- Joint pre-training text + image-text pairs (LAION, COYO, proprietary), 5% corpus
- Native multimodal: Ayrı image encoder + adapter değil, main Transformer'a entegre, tighter image-text reasoning

### 5.11 Training Methodology
- Corpus: ~15T token text + image-text, 60% web (CommonCrawl, C4) filtered deduplicated, 15% code (500+ lang), 10% academic papers & books (arXiv, PubMed, Gutenberg), 10% multilingual 60+ lang Chinese ağırlıklı, 5% image-text pairs
- Data cutoff Aug 2026, release Sep 10
- Tokenizer: V4 Flash/Pro ile aynı BPE 102400 vocab, 0.7 token/char English, 0.6 Chinese
- Optimizer: AdamW β1=0.9, β2=0.95, weight decay 0.1
- LR: Peak 3e-4 cosine decay 3e-5 over 15T tokens
- Batch: 4M tokens/step global
- Sequence length: Early 4K, mid 16K, final 5% 1M
- Mixed precision: BF16 activations, FP32 optimizer states
- Hardware: Tahmini 8192 H100 80GB 60 gün (resmi değil)
- Post-training: SFT 2M high-quality instruction-response 3 epoch lr 1e-5, RL PPO reward model 500K human preference, reasoning depth, hallucination azaltma, alignment

### 5.12 Benchmarks ve Maliyet
- GPQA Diamond 90.9, HLE w/tools 63.9 (+18.8 vs V4 Flash), DeepSWE v1.1 74.2 (+14.9 vs V4 Flash 54.4, matches Opus 5), Terminal-Bench 2.1 90.6 (beats Opus 5 89.1), Codeforces 3471
- Pricing off-peak: $0.15/M input cache miss, $0.003/M cache hit, $0.60/M output; peak: $0.30/M, $0.006/M, $1.20/M; peak hours 01:00-04:00 ve 06:00-10:00 UTC Mon-Fri
- V4 Pro'dan 3.3x ucuz ($1.98/M output off-peak), Claude Opus 4.8'den 42x ucuz ($25/M) comparable GPQA
- Inference: Global KV 890 bytes/token, 1M context 850MB vs 3.2GB, 3.7x throughput

### 5.13 Stratejik Çıkarımlar
- Architectural efficiency > parameter scaling - smarter beats bigger
- KV cache compression (CED + CSA2 + FP4 + SWA) long-context agentic workload'lar için serving cost'u dramatik düşürüyor
- Cache-hit charges agent cost'un büyük kısmı, compression doğrudan maliyeti kesiyor
- Open weights MIT ama 552B MoE deployment hala heavy, memory bandwidth bottleneck (expert weights fetch)
- Native multimodal + 1M context + 384K output = agent workloads için optimize (large repos, tool calls, images mixed with text)

---

## Özet - Öğrenilenler

Bu derin okuma sonrası:
- Transformers reposunun kod organizasyonunu, attention implementasyon varyantlarını (eager, SDPA, FlashAttention), KV cache yönetimini, modular code generation sistemini anladım
- AdamW'nin neden Adam'dan farklı olduğunu, decoupled weight decay'nin matematiksel ve Bayesian justification'ını, pratik etkilerini (hyperparam decoupling, generalization) öğrendim
- GaLore'un gradient low-rank property teorisini, SVD projeksiyonunu, multi-subspace switching ile full-rank learning'i nasıl koruduğunu, memory hesabını ve 7B modeli 24GB'de train etme başarısını kavradım
- Linear attention'un kernel trick ile O(N²)->O(N) nasıl düştüğünü, causal masking için cumulative sum trick'ini, RNN equivalence'ini, gated ve hybrid varyantlarını ve neden pure linear'ın yetersiz kalıp hybrid'in tercih edildiğini öğrendim
- DeepSeek v4.1 Flash'in CED, CSA2, FP4, SWA, mHC, MoE, Engram, DSpark gibi tüm bileşenlerini, nasıl 4x KV cache reduction ve 1/6 aktif parametre ile Pro'yu geçtiğini, agentic workload optimizasyonunu detaylıca anladım

Bu bilgilerle artık transformers mimarisi, optimizer'lar ve modern efficient attention teknikleri hakkında detaylı teknik tartışma yapabilir, kod okuyup modifiye edebilir, yeni mimari önerileri değerlendirebilirim.
