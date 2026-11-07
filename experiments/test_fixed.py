import torch, torch.nn as nn, torch.nn.functional as F, math

def phi(x):
    return F.elu(x) + 1

def linear_attn(Q,K,V):
    Qp=phi(Q); Kp=phi(K)
    B,H,N,D=Qp.shape
    _,_,_,M=V.shape
    S=torch.zeros(B,H,D,M)
    Z=torch.zeros(B,H,D)
    outs=[]
    for i in range(N):
        k=Kp[:,:,i,:]; v=V[:,:,i,:]
        S=S+k.unsqueeze(-1)*v.unsqueeze(-2)
        Z=Z+k
        q=Qp[:,:,i,:]
        num=torch.einsum('bhd,bhdm->bhm', q, S)
        den=torch.einsum('bhd,bhd->bh', q, Z).unsqueeze(-1)+1e-6
        outs.append(num/den)
    return torch.stack(outs,dim=2)

def delta_attn(Q,K,V,beta=0.05):
    Qp=phi(Q); Kp=phi(K)
    B,H,N,D=Qp.shape
    _,_,_,M=V.shape
    S=torch.zeros(B,H,D,M)
    outs=[]
    for i in range(N):
        k=Kp[:,:,i,:]; v=V[:,:,i,:]
        pred=torch.einsum('bhd,bhdm->bhm', k, S)
        err=v-pred
        # clip for stability
        err=torch.clamp(err, -1, 1)
        S=S+beta*k.unsqueeze(-1)*err.unsqueeze(-2)
        q=Qp[:,:,i,:]
        outs.append(torch.einsum('bhd,bhdm->bhm', q, S))
    return torch.stack(outs,dim=2)

class JormungandrTiny(nn.Module):
    def __init__(self, vocab=64, d=64, h=2):
        super().__init__()
        self.emb=nn.Embedding(vocab,d)
        self.q=nn.Linear(d,d); self.k=nn.Linear(d,d); self.v=nn.Linear(d,d)
        self.ffn=nn.Sequential(nn.Linear(d,d*2), nn.GELU(), nn.Linear(d*2,d))
        self.ln1=nn.LayerNorm(d); self.ln2=nn.LayerNorm(d)
        self.out=nn.Linear(d,vocab)
        self.h=h; self.dh=d//h
    def forward(self,x):
        B,N=x.shape
        h=self.emb(x)
        Q=self.q(h).view(B,N,self.h,self.dh).transpose(1,2)
        K=self.k(h).view(B,N,self.h,self.dh).transpose(1,2)
        V=self.v(h).view(B,N,self.h,self.dh).transpose(1,2)
        attn=delta_attn(Q,K,V,beta=0.05).transpose(1,2).contiguous().view(B,N,-1)
        h=self.ln1(h+attn)
        h=self.ln2(h+self.ffn(h))
        return self.out(h)

def copy_task(seq_len=128, batch=16, vocab=64):
    x=torch.randint(0,vocab,(batch,seq_len))
    y=torch.roll(x,-1,1)
    y[:,-1]=x[:,0]
    return x,y

def train(seq_len=128):
    torch.manual_seed(0)
    m=JormungandrTiny(vocab=64,d=64,h=2)
    opt=torch.optim.AdamW(m.parameters(),lr=2e-3)
    for step in range(80):
        x,y=copy_task(seq_len=seq_len,batch=8)
        logits=m(x)
        loss=F.cross_entropy(logits.view(-1,64), y.view(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.0)
        opt.step()
        if step%20==0:
            print(f"seq {seq_len} step {step} loss {loss.item():.3f}")
    with torch.no_grad():
        x,y=copy_task(seq_len=seq_len,batch=32)
        pred=m(x)[:,-1,:].argmax(-1)
        acc=(pred==y[:,-1]).float().mean().item()
        print(f"seq {seq_len} FINAL long-range acc {acc:.1%} loss {loss.item():.3f}")
        return acc

for L in [64,128,256,512]:
    train(L)
