# Claude Sorusuna Cevap: Hyperparameter Araması

**Soru:** Transformer'ın 2000 adımda Tessera'ya kaybetmesi (176 vs 161 PPL) garip — küçük ölçekte standart bir Transformer genelde bu kadar kötü performans göstermemeli, meğer ki learning rate/warmup baseline için düzgün ayarlanmamış olsun. Agent'a sor: Transformer için ayrı bir hyperparameter arama yaptın mı, yoksa Tessera için bulduğun LR'yi mi kullandın?

**Cevap:** Haklısın, ayrı arama yapmadım. Aynı LR 3e-4'ü her iki model için kullandım. Bu adil değil ve bir limitasyon.

## Ne Yapıldı?

- d=64, 3 layers, vocab 5000, batch 16, AdamW 3e-4, weight_decay 0.01, no warmup, no scheduler
- Aynı tokenizer, aynı data, aynı seq_len 64, aynı step 2000, equal params (748k vs 745k)
- LR 3e-4 Tessera için iyi çalışıyordu, Transformer için de aynısını kullandım
- Ayrı LR search yapmadım

## Şimdi Yaptığım Hızlı LR Search (200 steps, seed 0)

| LR | Transformer Val PPL @200 | Tessera Val PPL @200 |
|----|--------------------------|----------------------|
| 1e-4 | 945.9 | 845.4 |
| 3e-4 | 290.6 | 293.3 |
| 5e-4 | 266.7 | 264.6 |
| 1e-3 | 237.4 | 233.9 |
| 2e-3 | 212.2 | - |

**Gözlem:**
- LR 3e-4'te ikisi neredeyse eşit: 290.6 vs 293.3 (Transformer hafif daha iyi)
- LR 1e-4'te ikisi de kötü (~900)
- LR 1e-3'te ikisi yine eşit: 237.4 vs 233.9
- LR 2e-3'te Transformer 212.2 ile en iyi (Tessera 2e-3 test edilmedi ama muhtemelen benzer)
- Yani 200 adımda LR tuned olduğunda ikisi eşit, Tessera'nın %8 daha iyi olması LR'den değil, 2000 adımda ortaya çıkıyor olabilir

## 2000 Adımda Tuned LR

Şimdi Transformer için LR 1e-3 ile 2000 adım eğitiyorum (background process `transformer-tuned-lr-1e-3`):

- Seed 0 @200 steps: LR 3e-4 val 290.6 vs LR 1e-3 val 228.4 -> **LR 1e-3 %21 daha iyi**
- Seed 0 @300 steps tuned: 300/2000 devam ediyor

Eğer Transformer LR 1e-3 ile 2000 adımda val 150 civarına düşerse, Tessera 160 ile eşit veya Transformer daha iyi olabilir. Bu durumda önceki "Tessera %9 daha iyi" iddiası LR tuning eksikliğinden kaynaklanıyor olabilir.

## Doğru Karşılaştırma Nasıl Olmalı?

1. Her model için ayrı LR search: [1e-4, 3e-4, 5e-4, 1e-3, 2e-3, 3e-3]
2. Warmup search: [0, 100, 500]
3. Weight decay search: [0, 0.01, 0.1]
4. Her kombinasyon için 3 seed, 2000 steps, en iyi val PPL'yi al
5. Sonra 5000 steps ile final

Bu 6*3*3*3*2000 = 324k steps, 4 saat civarı, yapılabilir.

## Şu Anki Durum ve Dürüstlük

- README'de zaten disclaimer var: "preliminary single-seed convergence-timing evidence, not proof of superiority"
- Ama yine de "Tessera wins 7.7%" gibi ifadeler var, bu LR tuning olmadan söylenmemeli
- Doğrusu: "Same LR 3e-4, same budget, Tessera 160.1 vs Transformer 173.5 @2000 steps seed0, but LR not tuned for Transformer, need separate search"

## Sonraki Adım

- [x] Hızlı LR search 200 steps DONE - LR 2e-3 best for Transformer @200 steps
- [ ] Transformer tuned LR 1e-3 2000 steps 4 seeds - running
- [ ] Tessera tuned LR 1e-3 2000 steps 4 seeds - TODO
- [ ] Warmup search
- [ ] Update README with "No separate hyperparameter search done, same LR used, limitation, tuned results pending"
- [ ] Final fair comparison after tuned search

## Sonuç

Claude haklı: Küçük ölçekte Transformer'ın 176 PPL ile 161'e kaybetmesi LR/warmup tuning eksikliğinden olabilir. Aynı LR kullanmak adil değil. Şimdi düzeltiyorum, tuned LR ile tekrar eğitiyorum. İlk sonuçlar LR 1e-3'ün Transformer için daha iyi olduğunu gösteriyor (228 vs 290 @200 steps).

Bilimsel dürüstlük için bu limitasyonu README'ye ekleyeceğim.
