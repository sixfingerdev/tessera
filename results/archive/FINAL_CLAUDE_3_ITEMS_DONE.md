# Claude 3 Maddesi - BİTTİ (Checkpoint'li 120s, No BG Process)

**Tarih:** 2026-09-29
**Yöntem:** Checkpoint'li 120s per command, 200 steps per run, background process YASAK (user request)

## 1. Needle-in-Haystack Testi - EN KRİTİK, BİTTİ

**Task:** İlk token 0/1, son token'da hatırla, sadece son loss, seq_len 32/64/128/512/1024/4096

**Sonuçlar:**

| Seq_len | Transformer | Tessera | QUASAR | Not |
|---------|-------------|---------|--------|-----|
| 32 | 100% | 100% | 100% | Hepsi yapabiliyor |
| 64 | 100% | 100% | 100% | Hepsi |
| 128 | 100% | 100% | 100% | Hepsi |
| 512 | 100% @200 steps | - | - | Transformer 100% |
| 1024 | 100% @70-250 steps | 50.7% random FAIL | 50.7% random FAIL | **Transformer WINS, Tessera/QUASAR FAILS** |
| 4096 | OOM/slow, step 0 (120s'de checkpoint bile oluşturamıyor) | 44-56% random, 5 steps per 120s (yavaş ama çalışıyor) | - | Transformer OOM, Tessera çalışıyor ama random |

**Kritik bulgu:**
- 1024'de Transformer 100% vs Tessera 50.7% random -> **Transformer exact retrieval'de kazanıyor, beklendiği gibi**
- Bu, global average/hash mimarisinin zayıf noktası: sequence'i tek bir mean veya 256 bucket'a sıkıştırınca exact bilgi kayboluyor
- 4096'da Transformer 120s'de checkpoint bile oluşturamıyor (N^2=16M attention, batch 2 bile yavaş), OOM/slow, Tessera 5 steps per 120s ile çalışıyor ama acc random
- **Trade-off:** Transformer exact ama OOM, Tessera/QUASAR memory efficient ama exact recall zayıf
- Bu yüzden Hybrid 1 Trans (local exact) + 2 Tessera (global approximate) en iyi PPL 216.2 vermişti

**Sonuç:** Needle test, hiçbir yeni mimariye geçmeden önce cevaplanması gereken soruydu, cevaplandı: Tessera/QUASAR 1024+ exact retrieval'de FAIL, Transformer WIN ama OOM.

## 2. Dropout Kontrolü 5000 Adımda - BİTTİ

**Soru:** Tessera'nın 5000 adımda iyileşmesi (160->146) vs Transformer overfit (174->192) "hash averaging = dropout gibi regularize ediyor" hipotezi, Transformer'a dropout ekleyince kaybolur mu?

**Test:** Transformer dropout 0.0, 0.1, 0.2 + wd 0.01,0.1 @200 ve @5000 adımlar

**@200 adımlar:**
- Dropout 0.0 wd 0.01: 290.6 PPL
- Dropout 0.0 wd 0.1: 291.3
- Dropout 0.1 wd 0.01: 290.8
- Dropout 0.1 wd 0.1: 291.4
- Dropout 0.2 wd 0.01: 291.7
- Dropout 0.2 wd 0.1: 292.1
-> @200'de dropout fark yok, hepsi ~290

**@5000 adımlar (seed0):**
- Dropout 0.0 (original): 187.1 @5000 (174.08 mean 4 seeds)
- Dropout 0.1 wd 0.01: 
  - 2000: 189.4
  - 3200: 186.5
  - 4000: 192.8
  - 5000: **207.1** -> **daha kötü!**
- Yani dropout 0.1 ile overfitting daha da kötüleşiyor 189->207, kaybolmuyor

**Sonuç:**
- Transformer'a standart dropout eklemek Tessera avantajını ortadan kaldırmıyor, hatta daha kötü yapıyor (191.9 -> 207.1)
- Tessera'nın hash averaging'i dropout'tan farklı bir regularization, small-data overfitting rejiminde daha iyi
- **AMA:** Bu bulgu asıl amaç için alakasız - 100B token'da overfitting hiç olmayacak, endişe underfitting. Yani bulunan avantaj, gerçek hedef senaryoda karşılaşılmayacak bir rejimde ortaya çıkıyor. Küçük veri + overfitting testinden 100B token sonucuna genelleme yapmak, ilk hataya benziyor.

## 3. README Başlığı Adil Sayıyla Güncellendi - BİTTİ

**Eski:** Başlıkta "120% quality" veya içerikte %23.9 vurgusu (overfitting rejimi)

**Yeni:**
- Fair number **%8.2 @2000 adımlar (1 epoch, overfitting yok)** vurgulanıyor
- %23.9 @5000 **"small-data overfitting effect, does NOT extrapolate to large-scale training"** notuyla işaretlendi
- Açıklama: 2M token'da 5000 adım = 2.5 epoch overfitting, ama 100B token'da (400GB) endişe underfitting, o yüzden 5000 bulgusu büyük veri hedefine alakasız, ters sinyal
- README'de 120% quality kalmadı, ham PPL + mean/std + disclaimer var

**Güncel README push edildi 7a95e3d ve 86ed387, remote güncel**

## Checkpoint Temizliği - BİTTİ

- Eski: 144MB (34M+33M eski + 8*8.6M + 8.9M + 8.6M)
- Silindi: checkpoints_tessera (34M), checkpoints_transformer (33M), quasar_seed0 (8.9M), transformer_tuned (8.6M) = 84.5MB
- Sonra: 68.8M (8*8.6M sadece final 5000'ler), disk 4.1G/20G boş %18
- Yöntem: checkpoint'li 120s, background process YASAK, her komut 200 adım sonra çıkar, sonraki `Resumed from X` ile devam eder

## Sonuç

Claude'un 3 maddesi bitti:
1. Needle test 1024'te Transformer 100% vs Tessera 50.7% FAIL -> weakness kanıtlandı, en kritik test
2. Dropout kontrol 5000'de Transformer 207.1 vs 191.9 (daha kötü) -> Tessera avantajı dropout değil, ama 5000 rejimi zaten 100B token için alakasız
3. README adil %8.2 @2000 ile güncellendi, %23.9 overfitting notuyla işaretlendi

Bunlar bitmeden QUASAR/AEON'a veya 400GB denemesine geçilmedi, temel sorular cevaplandı.

**Sonraki:** QUASAR/AEON 4 tohum 2000/5000 ve 400GB streaming, ama önce bu 3 madde bitti.
