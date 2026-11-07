"""
Length-generalization probe for Lodestar on the random-position needle task.
Train at L=1024 (short, 120 steps), then ZERO-SHOT evaluate soft/hard at
512 / 1024 / 2048 / 4096 / 8192.
Honest purpose: if hard-exactness is STRUCTURAL (content-quantized routing that
isolates the needle by content, independent of N), accuracy should stay 100%
beyond the training length. If the model memorized length-specific behavior,
it will collapse.
"""
import torch, math, json
from lodestar import LodestarModel, gen_needle, evaluate

def main():
    torch.manual_seed(6); torch.manual_seed(6)
    import random; random.seed(6)
    L, steps = 1024, 120
    model = LodestarModel(vocab=10, d=64, layers=3, slots=16)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    for step in range(1, steps + 1):
        model.train()
        x, y = gen_needle(4, L)
        loss = torch.nn.functional.cross_entropy(model(x)[:, -1, :], y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 30 == 0:
            print(f"train step {step} loss {loss.item():.4f}", flush=True)
    torch.save(model.state_dict(), "/tmp/lode_needle_lenprobe.pt")
    model.eval()
    results = {}
    with torch.no_grad():
        for Le in [512, 1024, 2048, 4096, 8192]:
            batch = 8 if Le <= 2048 else (4 if Le == 4096 else 2)
            accs = evaluate(model, gen_needle, batch=batch, L=Le, n_batches=12)
            results[Le] = accs
            print(f"L={Le:5d} (train 1024, zero-shot): soft {accs['soft']:.1%} "
                  f"hardread {accs['hardread']:.1%} hard {accs['hard']:.1%}", flush=True)
    print("[lenprobe-json] " + json.dumps({str(k): v for k, v in results.items()}), flush=True)

if __name__ == "__main__":
    main()
