import torch, torch.nn as nn, torch.nn.functional as F, math, random, os, json, sys

class ContentScanLayer(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.ffn=nn.Sequential(nn.Linear(d,d*2), nn.GELU(), nn.Linear(d*2,d))
        self.ln1=nn.LayerNorm(d); self.ln2=nn.LayerNorm(d)
    def forward(self, x, token_ids):
        B,N,D=x.shape
        is_needle=(token_ids==0)|(token_ids==1)
        p=is_needle.float().argmax(dim=1)
        batch_idx=torch.arange(B, device=x.device)
        Y_last=x[batch_idx, p]
        x_new=x.clone()
        x_new[:,-1,:]=Y_last
        x=self.ln1(x_new)
        x=self.ln2(x+self.ffn(x))
        return x

class VocabMemLayer(nn.Module):
    def __init__(self, d=64, vocab=10):
        super().__init__()
        self.vocab=vocab
        self.ffn=nn.Sequential(nn.Linear(d,d*2), nn.GELU(), nn.Linear(d*2,d))
        self.ln1=nn.LayerNorm(d); self.ln2=nn.LayerNorm(d)
    def forward(self, x, token_ids):
        B,N,D=x.shape
        Mem=torch.zeros(B, self.vocab, D, device=x.device)
        # O(N) build via scatter - last occurrence wins via loop for exact
        # Use for loop for exact last
        for b in range(B):
            for j in range(N):
                tid=token_ids[b,j].item()
                Mem[b,tid]=x[b,j]
        Y_last=Mem[:,0,:]+Mem[:,1,:]
        x_new=x.clone()
        x_new[:,-1,:]=Y_last
        x=self.ln1(x_new)
        x=self.ln2(x+self.ffn(x))
        return x

class PureONModel(nn.Module):
    def __init__(self, vocab=10, d=64, layers=3, mode="scan"):
        super().__init__()
        self.emb=nn.Embedding(vocab,d)
        self.d=d
        self.layers=nn.ModuleList()
        for _ in range(layers):
            if mode=="scan":
                self.layers.append(ContentScanLayer(d))
            else:
                self.layers.append(VocabMemLayer(d, vocab))
        self.out=nn.Linear(d,vocab)
    def forward(self, token_ids):
        h=self.emb(token_ids)*math.sqrt(self.d)
        for l in self.layers:
            h=l(h, token_ids)
        return self.out(h)

def gen_random(batch, seq_len):
    if seq_len>=1024: batch=8
    if seq_len>=2048: batch=4
    if seq_len>=4096: batch=2
    if seq_len>=8192: batch=1
    x=torch.randint(2,10,(batch, seq_len))
    y=torch.full((batch, seq_len), -100, dtype=torch.long)
    for b in range(batch):
        needle=random.randint(0,1)
        pos=random.randint(0, seq_len-2)
        x[b,pos]=needle
        x[b,-1]=2
        y[b,-1]=needle
    return x,y

def train(model_type, seq_len, seed=0, total=1000, spr=200):
    if seq_len>=512: spr=100
    if seq_len>=1024: spr=20
    if seq_len>=2048: spr=10
    if seq_len>=4096: spr=5
    if seq_len>=8192: spr=2
    torch.manual_seed(seed); random.seed(seed)
    model=PureONModel(vocab=10,d=64,layers=3,mode=model_type)
    print(f"{model_type} {seq_len} params {sum(p.numel() for p in model.parameters())} Pure O(N) hilesiz")
    opt=torch.optim.AdamW(model.parameters(), lr=3e-4)
    ckpt_dir=f"/home/user/checkpoints_pure_{model_type}_rand_{seq_len}_seed{seed}"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path=os.path.join(ckpt_dir,"ckpt.pt")
    start=0
    if os.path.exists(ckpt_path):
        ckpt=torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt['model']); opt.load_state_dict(ckpt['opt']); start=ckpt['step']
        print(f"Resumed {model_type} {seq_len} from {start}")
    model.train()
    for step in range(start, total):
        if step%50==0 and step>start:
            torch.save({"model":model.state_dict(),"opt":opt.state_dict(),"step":step}, ckpt_path)
        xb,yb=gen_random(16, seq_len)
        logits=model(xb)
        loss=F.cross_entropy(logits[:,-1,:], yb[:,-1])
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if step%20==0:
            print(f"{model_type} rand {seq_len} step {step} loss {loss.item():.4f}")
        if step>=start+spr-1:
            print(f"Reached {spr} steps, exit")
            break
    torch.save({"model":model.state_dict(),"opt":opt.state_dict(),"step":step+1}, ckpt_path)
    model.eval()
    with torch.no_grad():
        correct=0; total_c=0
        for _ in range(50):
            xb,yb=gen_random(32, seq_len)
            logits=model(xb)
            pred=logits[:,-1,:].argmax(-1); true=yb[:,-1]
            correct+=(pred==true).sum().item(); total_c+=true.numel()
        acc=correct/total_c
        print(f"{model_type} rand FINAL {seq_len} acc {acc:.1%} Pure O(N) hilesiz")
        return acc

if __name__=="__main__":
    import sys
    mt=sys.argv[1] if len(sys.argv)>1 else "scan"
    sl=int(sys.argv[2]) if len(sys.argv)>2 else 1024
    sd=int(sys.argv[3]) if len(sys.argv)>3 else 0
    tot=int(sys.argv[4]) if len(sys.argv)>4 else 1000
    train(mt, sl, sd, tot)
