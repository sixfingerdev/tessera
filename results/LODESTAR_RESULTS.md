# LODESTAR — Learned Bucket-Routed Causal Associative Memory
## Dürüst deneysel rapor (2026-09-30)

**Bir cümlelik özet:** Kod içinde token ID'si ya da pozisyon bilgisi **olmayan**, tamamen öğrenilmiş bir
slot-yönlendirmeli bellek, rastgele-pozisyonlu needle testini %100 (hard discrete modda) çözüyor,
**8192 uzunluğa kadar sıfır-shot genelliyor** ve ID-permütasyonu altında dayanıklı —
ama bu, genel LM'in açık problemini **çözmüyor** ve bu rapor hiçbir yerde öyle bir iddia içermiyor.

---

## 0. Bu deney neden var? (Claude'un 2. tur eleştirisi sonrası)

Önceki turda iki "çözüm" geri çekildi: Content-Scan ve Vocab-Indexed Memory, `token_ids == 0 | 1`
diye kod içine gömülü cevap anahtarıydı (hile, öğrenme yok). Gated Pointer pozisyon hilesiydi.
Claude'un test sorusu: **"Bu mimari, needle testinin spesifik yapısını (hangi token'lar özel,
needle hangi pozisyonda) bilmeden de çalışır mıydı?"**

LODESTAR bu soruya karşı tasarlandı:
- Model kodunda (`lodestar.py` → `LodestarLayer`, `LodestarModel`) **hiçbir yerde** token ID'si,
  pozisyon kuralı, "0/1 özel", "son token query" gibi bir bilgi yok. Kod sadece [B,N] token IDs görür.
- Özel olup olmadığı **iki bağımsız sınama** ile test edildi (aşağıda: ID-permütasyon probu ve
  uzunluk-genellemesi probu). İkisi de geçti.

## 1. Mekanizma

Tek katman (tamamı öğrenilmiş, O(N·K·D), nedensel):

```
yazma adresi:  Aw[j] = softmax(x_j W_w / τ_w)        [K]  "değerimi hangi slota koyarım"
okuma adresi:  Ar[i] = softmax(x_i W_r / τ_r)        [K]  "hangi slota bakarım"   (W_w ≠ W_r!)
slot içeriği:  S[k,i] = Σ_{j≤i} Aw[j,k]·x_j          (nedensel prefix toplam)
               C[k,i] = Σ_{j≤i} Aw[j,k]
               slot[k,i] = S[k,i] / (C[k,i]+ε)       (koşan slot ortalaması)
okuma:         y_i = W_o · Σ_k Ar[i,k] · slot[k,i]   (yalnızca j ≤ i → CAUSAL)
x = LN(x + gate·y);  x = LN(x + FFN(x))
```

- Tessera'nın aksine **yazma ve okuma adresleri farklı projeksiyonlar** — sorgu, aradığı içeriğin
  slotuna adreslenebilir. (Tessera'da `gather(bucket_means, bucket_ids)` her tokena yalnız kendi
  kovasını verir; needle'ı taşıması yapısal olarak imkânsızdı — %50.7 şans seviyesi bundandır.)
- Değerler ortalama ile değil **izole yazılıp** (aşağıda kanıtı var) hard modda tek adresten
  geri okunabildiği için exact mümkün.
- Karmaşıklık: katman başına zaman O(N·K·D), state K·D. K=16-24 sabit → **subquadratic**.
  Ölçüm (CPU, batch 1, d=64, K=16): N=1024: 0.021s, N=4096: 0.114s, N=16384: 0.560s,
  N=65536: 2.416s (doğrusal; aynı N'de yoğun N×N attention matrisi 4/64/1024/16384 MB olurdu).

## 2. Sonuçlar

### 2.1 Rastgele-pozisyonlu needle (repo'nun gerçek testi; vocab 10, d=64, 3 katman, K=16)

Eğitim: 400 adım, lr 1e-3, batch 4, değerlendirme 800 örnek.

| Model | L | seed | soft | hardread | hard | not |
|---|---|---|---|---|---|---|
| **LODESTAR** | 1024 | 0 | **100.0%** | – | **100.0%** | loss 0.0013 |
| **LODESTAR** | 1024 | 1 | **100.0%** | 100.0% | **100.0%** | |
| **LODESTAR** | 1024 | 2 | **99.9%** | – | **100.0%** | |
| **LODESTAR** | 1024 | 5 (kısa, 120 adım) | 100% | 100% | 100% | uzunluk problası için |
| Transformer | 1024 | – | 100% @70 | | | O(N²), 4096'da OOM |
| Tessera / QUASAR / Linear | 1024 | – | 50.7% FAIL | | | yapısal: kendi kovasını okuyor |
| Gated Pointer | 1024 | – | 100% sabit-pos, **%50 FAIL random** | | | pozisyon hilesi |
| Content-Scan / VocabMem | 1024 | – | "100%" | | | **GERİ ÇEKİLDİ** — token-ID hilesi |

**Uzunluk genellemesi (sıfır-shot; yalnızca L=1024'te eğitildi, 120 adım, seed 6):**

| Değerlendirme L | soft | hardread | hard |
|---|---|---|---|
| 512 | 100% | 100% | 99.0% |
| 1024 | 99.0% | 100% | 100% |
| 2048 | **100%** | **100%** | **100%** |
| 4096 | **100%** | **100%** | **100%** |
| 8192 | **100%** | **100%** | **100%** |

Eğitim uzunluğunun **8 katında** bile hard-exact %100. Exactness ezber değil, **yapısal**:
iğne, içeriğiyle anahtarlanmış slota tek başına yazılıyor (load=1) — bu N'den bağımsız.

### 2.2 Öğrenilen devre (dürüst tanı: tasarladığım şey bu değildi)

Kayıtlı model üzerinde katman katman iz (hard mod, gerçek iğne konumu):
- **Katman 0:** iğne (tok 0 → slot 11, tok 1 → slot 6; tüm dizilerde tutarlı) **tek başına**
  slota yazılıyor (load=1) — yani içerik-nicemlemeli izolasyon, kelimesi kelimesine kopya.
  Sorgunun okuduğu slot boş (y=0) — katman 0 taşıma yapmıyor, **izole ediyor**.
- **Katman 1:** iğnenin residual'ı kendi değerini aldı; 148 pozisyon iğnenin slotuna yöneliyor,
  iğne bilgisi bu pozisyonların hidden state'lerine **difüzyonla** yayılıyor; sorgunun slotuna
  yazan 30 pozisyon var.
- **Katman 2:** sorgu 30-yazarlı slotu okuyup bilgiyi topluyor; çıkış katmanı decode ediyor.
- Tek-katmanlık "iğne slotta + sorgu aynı slotu okuyor" eşleşmesi **hiçbir katmanda yok**
  (route_exact=0%): devre **çok-sıçramalı içerik-anahtarlı bir yönlendirme grafiği**.
  Uçtan uca hard %100 olduğu için bu bir başarısızlık değil, **bulunan çözümün gerçek şekli**.

### 2.3 Anti-hile probu 1 — ID permütasyonu

Aynı kod, farklı roller: needle ∈ {5,7}, query = 3, filler'lar geri kalanlar. Sıfırdan eğitim:

| | L | seed | soft | hard |
|---|---|---|---|---|
| **LODESTAR** | 1024 | 0 | **100.0%** | **100.0%** |

Kodda ID önyargısı olmasaydı bu geçemezdi; geçti → **mimari görev yapısını bilmiyor, öğreniyor**.

### 2.4 Anti-hile probu 2 — rol-karıştırmalı ilişkisel hatırlama (assoc)

ID-permütasyonu hâlâ "rolller sabit" sayılır (0/1 hep iğne *türünde*). Daha sert test:
her dizide roller **yeniden karıştırılıyor** (marker ve value token'ları {0,1,2,3} havuzundan
dizi başına rastgele; `m …gap… v …fillers… q=m`, cevap = v). Token kimliği rolü belirleyemez;
değerin adresini **bağlamdan** (öncesindeki marker) öğrenmesi gerekir. Mekanizmaya bunun için
**nedensel yerel bağlam** eklendi (24 pencere ortalaması + lineer karışım, yalnız router girdisinde;
O(N·w·d)):

| Görev | L | adım | soft | hardread | hard |
|---|---|---|---|---|---|
| assoc1 (tek bağ) | 64 | 650 (zirve) | **98.8%** | – | 44.4% |
| assoc1 | 64 | 1200 final | 79.8% | – | 38.1% (agresif tavlıma çözümü bozdu) |
| assoc1 | 256 | 900 (zirve) | **96.2%** | – | 55.6% |
| assoc1 | 256 | 1500 final | 68.8% | – | 49.4% (tavlama soft'u da düşürdü) |
| assoc2 (çift bağ) | 256 | 2500 final | **93.4%** | **94.5%** | 53.1% |

**Hile denetimi (otomatik):** `LodestarLayer`/`LodestarModel` kodunda token-ID veya pozisyon
referansı yoktur (grep denetimi temiz); modelin gördüğü tek girdi `[B,N]` token dizisidir.

**assoc2 kritik tanı:** `hardread` (okuma discrete, hafıza graded) %94.5 ≈ soft %93.4 — yani
**discrete okuma baş başına yeterli**; tam-hard'a düşüş (%53.1) **discrete yazıdan** kaynaklanıyor:
graded slot kütleleri argmax'a zorlanınca bilgi kayboluyor. Dönüşümün tamamı için yazı tarafında
yeni bir fikir gerekiyor (bu bütçede bulunamadı — bkz. negatif sonuçlar).

**Dürüst okuma:** bağlama (binding) yumuşak modda öğreniliyor (assoc2 %93.4; assoc1 zirve %98.8)
ama bu eğitim bütçesinde **tam discrete yönlendirmeye dönüşmüyor** — hard ~%50-59. Yani:
- İçerik-kararlı görevlerde (needle): Lodestar **hard-exact** (yapısal, uzunluk-genel).
- Rol-bağımlı görevlerde: Lodestar şu an **yumuşak (yaklaşıksal-ama-yüksek)** bir ilişkisel bellek;
  discrete Exact'e dönüşüm açık bir optimizasyon problemi olarak kalıyor.

### 2.5 Negatif sonuçlar (aynı derecede önemli)

1. **Sert yönlendirmeyle sıfırdan başlamak (STE) çöktü:** assoc1'de 250 adımda %0.0 —
   rastgele discrete başlangıç bilgiyi taşımıyor; yumuşak faz olmadan optimizasyon başlamıyor.
2. **Keskin okuma olmadan yumuşak eğitim plato:** τ_r=0.5 iken assoc1 %26-35'te (şans %25) takılıyordu;
   τ_r=0.1 (sorgu baştan tek slot seçer) öğrenmeyi açtı. Okuma-yazma eş-güvence (chicken-egg)
   probleminin kilidi okuma tarafının erken keskinleşmesi oldu.
3. **Agresif τ_w tavlaması (0.4'ten başlayıp 0.15'e) öğrenilen yumuşak çözümü bozdu:**
   %98.8 → %79.8. Dağıtık süperpozisyon çözümü discrete'e zorlama bilgiyi eziyor. Daha yumuşak
   tavlama (0.6'dan, taban 0.25) hard'ı kısmen iyileştirdi (45→56%) ama çözdü değil.
4. **Geçersiz koşu:** needle 4096 ilk deneme yanlış CLI ile K=2 slotla koşuldu (%51 — anlamlı değil,
   atıldı). Yerine uzunluk-genellemesi probundaki sıfır-shot 4096/8192 sonuçları kullanıldı
   (daha güçlü kanıt: eğitimsiz genelleme).
5. Route_exact diyagnostiğinin ilk sürümü yalnız katman 0'a bakıyordu; multi-hop devreyi
   görünce düzeltildi ve gerçek durum rapora yazıldı (2.2).
6. Geç kırılım: assoc görevlerinde kırılım ~700-2000 adım arasında oluyor; kısa koşular
   (%26-35'te plato) yanlışlıkla "öğrenilemez" sonucu verirdi. Rapordaki bütün assoc
   sayıları kırılım sonrası değerlerdir.

## 3. Bu ne DEĞİLdir (açık problemin dürüst durumу)

- LODESTAR genel LM için O(N²)'den ucuz **exact** retrieval çözümü **DEĞİLDİR** ve bu iddia yoktur.
- Kapasite arguably: K slot, aynı anda en fazla K bağımsız canlı bağ taşır (pigeonhole).
  Needle 1 bağ ister → K=16 bol. Genel LM'de N canlı bağ → K=Θ(N) gerekir → O(N·K)=O(N²).
  Yani **sabit K ile subquadratic + sınırsız exact aynı anda imkânsız** — bu mühendislik değil,
  kapasite sınırı. Genel LM açık problemi Claude'un çerçevelediği gibi **açık**.
- Worst-case'te subquadratic exact attention'ın imkânsızlığına dair güçlü teorik kanıtlar var
  (quadratic-time completeness). Gerçekçi hedef: doğal dağılımlarda öğrenilmiş indeks +
  doğrulama; bu deney o programın en basit "öğrenilmiş ters-indeks" veri noktasıdır.
- Akraba önceki çalışmalar (dürüstlük): product-key memory (Lample 2019), Reformer LSH
  attention, Routing Transformer / clustered attention, Set Transformer inducing points,
  Linformer/Nyströmformer/Luna bottleneck'leri, linear attention (Lodestar ≈ anahtarları K
  öğrenilmiş prototipe nicemlenmiş, yazma/okuma uzayları ayrık linear attention).
  Buradaki katkı bir "yeni fizik" değil: **ayrık yazma/okuma adres ayrımı + nedensel slot
  ortalamaları + anti-hile test protokolü** ve negatif/pozitif bulguların dürüst dökümü.

## 4. Tekrarlama

```bash
pip install torch
python lodestar.py needle 1024 0 400 1e-3        # %100 soft+hard (~8 dk CPU)
python lodestar.py needle_perm 1024 0 400 1e-3   # ID-perm probu, %100
python lodestar.py assoc1 64 0 1200 1e-3 24 16   # rol-karıştırma probu
python lodestar.py assoc2 256 0 2500 1e-3 24 16  # çift bağ
python lodestar_eval_len.py                      # 1024 eğit → 8192 sıfır-shot
```

Model kodunda token ID / pozisyon sabiti **yoktur**; görev tanımları (`gen_*`) tabii ki görevi
tanımlar — modelin gördüğü tek şey token dizileridir.

## 4b. Kapasite ölçümü — genel-LM rejimine doğru (assocM, çok-sorgulu)

**Kullanıcının sorusu:** "son token için değil, genel LM için exact retrieval?" LODESTAR'da okuma
zaten **her pozisyonda** yapılıyor (tek geçişte tüm pozisyonlar, O(N·K)); "yalnız son token"
sınırlaması needle *değerlendirmesinden* geliyordu, mekanizmadan değil. Genel LM ile aradaki
fark **kapasite**: exact, canlı bağ sayısı ≤ K iken mümkün. Bunun deneysel haritası için
tek dizide M bağ + M ayrı sorgu pozisyonu (çok-sorgulu, proxy) test edildi:

| M (canlı öğe=2M) | K | adım | soft | hardread | hard | şans |
|---|---|---|---|---|---|---|
| 2 | 24 | 2500 | **93.4%** | **94.5%** | 53.1% | 25% |
| 4 | 24 | 3000 (kırılmadan kesildi) | %67 (zirve %72) | 69.1% | 27.3% | 25% |
| 8 | 32 | 3000 (kırılmadan kesildi) | 33.6% (tımanıyor) | 34.3% | 9.9% | 12.5% |

**Dürüst bulgu:** Kapasite (K ≥ 2M) yeterli olsa bile **kırılım adım sayısı M ile dik artsılı
büyüyor** (M=2: ~2000 adım; M=4: >3000; M=8: ≫). Bu, kapasite sınırının kendisinden önce gelen
**optimizasyon zorluğu** sorunudur; iki koşu bütçe içinde kırılmadan kesildi (kesme noktasında
loss hâlâ düşüyordu). Yani "genel LM'e doğru kapasite merdiveni" şu an deneysel olarak
M=2'de çözülmüş, M≥4'te **açık** durumda — ne mimari kanıtı ne çürütüsü var, sadece
eğitim bütçesi yetersiz.

**Genel LM'e tam geçişin matematiği (değişmedi):** gerçek dilde canlı öğe kümesi ~vocab ölçeğinde;
exact için K=Θ(canlı öğe) gerekir → O(N·K), K büyükken O(N²)'yi geçer. "Exact + subquadratic +
genel" üçgeninin üçü aynı anda — teorik gerekçelerle (pigeonhole + quadratic-time completeness)
iice kapalı görünüyor. Açık problemstatement aynen geçerli.

## 5. Sonraki adımlar (öncelik sırasıyla, iddiasız)

1. assoc hard-discrete dönüşümü: yumuşak→sert tavlamayı çözmeden "exact associative routing"
   iddiası getirilemez. (Denenen: STE-from-scratch ✗, agresif anneal ✗, keskin-okuma ✓-soft)
2. K'nın canlı bağ sayısıyla ölçeklenme eğrisi (assoc2, assoc4, …): subquadratic bölgede
   kapasite sınırının deneysel haritası.
3. Gerçek dil: WikiText-2 PPL + needle karışık görev; K·D state'in dil modellemede yeterliliği.
4. Genel LM açık problemi: Θ(N) index + o(N) sorgu ile doğal dağılımda exact retrieval —
   hâlâ açık, hâlâ kolay cevabı yok.
