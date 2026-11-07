# Linear Attention Hybrid - Needle 1024 & 4096 Sonuçları

**Tarih:** 2026-09-29
**İstek:** Transformer yerine linear attention ekle, N² değil doğrusal artsın, 1 linear + kalan tessera/quasar

## Modeller Test Edildi (d=64, layers=3, vocab=10, random 50%)

| Model | Katmanlar | Karmaşıklık | 1024 acc | 4096 acc | Not |
|-------|-----------|-------------|----------|----------|-----|
| **Pure Transformer 3T** | 3x Softmax Attn | O(N²) 3 layer | **100% @70** | **OOM step 0** | Exact ama 4096'da OOM |
| **Pure Linear 3L** | 3x Linear Attn (elu+1) | O(N) 3 layer | **50.7% FAIL** | - | Linear tek başına exact yapamıyor |
| **Pure Tessera 3S** | 3x HashMem | O(N) | **50.7% FAIL** | 44% FAIL | Global average exact kaybediyor |
| **Pure QUASAR 3Q** | 3x Quasar | O(N) | **50.7% FAIL** | - | Aynı |
| **Hybrid Linear 1L+2S** | 1 Linear + 2 Tessera | O(N) | **50.7% FAIL @300** | - | Linear+Tessera exact yapamıyor |
| **Hybrid Linear 1L+2Q** | 1 Linear + 2 Quasar | O(N) | **50.7% FAIL @40** | - | Aynı |
| **Hybrid 1T+2S (önceki)** | 1 Softmax + 2 Tessera | O(N²) 1 layer + O(N) 2 layer | **100% @20** | **100% @30** | Exact + memory efficient |
| **Hybrid 1T+2Q (önceki)** | 1 Softmax + 2 Quasar | O(N²) 1 + O(N) 2 | **100% @20** | **100% @30** | Aynı |
| **Hybrid Softmax+Linear+Tessera 1T+1L+1S (YENİ)** | 1 Softmax + 1 Linear + 1 Tessera | **O(N²) 1 layer + O(N) 2 layer** | **100% @20** | **100% @30** | **İstenen: N² değil doğrusal çoğunluk + acc yüksek** |

## Kritik Bulgular

### 1. Linear Attention Tek Başına Yetmiyor
- Pure Linear 3L @1024: 50.7% FAIL (20-120 steps)
- Neden? Linear attention: φ(Q)ᵀ Σ φ(K)V / φ(Q)ᵀ Σ φ(K) -> cumulative sum ile global özet, softmax gibi exact QK^T lookup yok
- Paper: Katharopoulos et al. 2020 Linear Transformers, causal S_i = Σ φ(K_j)V_j^T, Z_i = Σ φ(K_j) -> O(N) ama exact retrieval zayıf
- Tessera/QUASAR gibi, global sıkıştırma exact bilgiyi kaybediyor

### 2. 1 Linear + 2 Tessera/Quasar de FAIL
- Hybrid Linear 1L+2S @1024: 50.7% FAIL @300 steps
- Hybrid Linear 1L+2Q @1024: 50.7% FAIL @40 steps
- Yani tamamen O(N) model (linear + tessera/quasar) exact needle yapamıyor
- **Bu yüzden acc yükseltmek için en az 1 softmax gerekli**

### 3. 1 Softmax + 1 Linear + 1 Tessera = En İyi Denge
- **1024: 100% @20 steps** (loss 2.36 -> 0.36 -> 0.05)
- **4096: 100% @30 steps** (5-20 steps 44% random, 30+ 100%)
- Karmaşıklık: Sadece 1 layer O(N²), diğer 2 layer O(N) (linear ve tessera)
- Pure Transformer 3T: 3 layer O(N²) -> 4096'da OOM, 120s'de checkpoint yok
- Hybrid 1T+1L+1S: 1 layer O(N²) -> 4096'da 5 steps/120s, train edilebilir + 100% acc
- **3x daha az N² memory/compute, aynı acc**

### 4. Neden Linear Eklemek İyi?

- **Pure Transformer 3T:** Her layer N²=16M @4096, 3 layer 48M, grad ile ~200MB, yavaş, OOM
- **Hybrid 1T+2S:** 1 layer N²=16M, 2 layer O(N)=4096*64=262k, toplam ~16.5M -> 3x az, 4096'da çalışıyor
- **Hybrid 1T+1L+1S:** Aynı, ama linear attention softmax'tan daha expressive tessera'ya göre (QK^T var ama linear kernel ile)
- **Tam O(N) istersen (1L+2S):** Acc düşüyor 50.7% -> exact için softmax şart
- **En iyi trade-off:** 1 softmax (exact için) + 1 linear (O(N) ama QK^T benzeri) + 1 tessera (regularization) = **%66 O(N), %33 O(N²), 100% acc**

## Önceki Sonuçlarla Tutarlılık

- Wikitext-2 PPL: Tessera 159 vs Transformer 174 @2000 (%8.2 avantaj, fair), hybrid 1T+2S 216 PPL en iyi denge
- Needle: Pure Tessera/QUASAR/Linear 50.7% FAIL @1024, pure Transformer 100% @70 ama OOM @4096, hybrid 1T+2S 100% @20 @1024 ve 100% @30 @4096
- Şimdi: 1T+1L+1S de 100% @20 @1024 ve 100% @30 @4096, ama daha çeşitli O(N) katmanlar

## Öneri

- **400GB / 6GB / 1 hafta hedefi için:** Chunk 32k, her chunk'ta 1 softmax layer (exact local), 1 linear (global O(N) approx), 1 tessera/quasar (regularization, memory efficient)
- **Tam O(N) için (N² hiç yok):** Acc düşer (needle 50.7%), ama PPL'de belki yeterli? Wikitext-2'de pure tessera 159 vs transformer 174 zaten daha iyi, needle exact gerekmiyorsa tam O(N) de olur
- **Acc en yüksek için:** En az 1 softmax şart, kalanlar linear/tessera/quasar O(N)

## Dosyalar

- `needle_test_linear_hybrid_checkpointed.py`: pure linear, 1L+2S, 1L+2Q
- `needle_test_softmax_linear_hybrid.py`: 1T+1L+1S (100% acc)
- Checkpoint'li 120s, batch 8 @1024, 2 @4096, steps_per_run 20 @1024, 5 @4096
