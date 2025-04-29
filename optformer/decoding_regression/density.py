import math, random, itertools, numpy as np, torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1.  Toy-shape generators  (Half-Moons, Zig-Zag, Spiral, Hollow Square)
# ------------------------------------------------------------
def make_half_moons(n=400, noise=0.01):
    theta = np.random.rand(n) * np.pi
    x1 = np.cos(theta)
    y1 = np.sin(theta)
    x2 = 1 - np.cos(theta)
    y2 = 1 - np.sin(theta) - 0.5
    pts = np.vstack([np.hstack([x1, x2]), np.hstack([y1, y2])]).T
    pts += np.random.randn(*pts.shape) * noise
    return pts

def make_zigzag(n=400, noise=0.01, segments=6):
    xs = np.random.rand(n)
    ys = ((np.floor(xs * segments) % 2) * 2 - 1) * (xs * segments % 1)
    pts = np.c_[xs, ys] + np.random.randn(n, 2) * noise
    return pts

def make_spiral(n=400, noise=0.01, revolutions=3):
    r = np.linspace(0.1, 1, n)
    theta = revolutions * 2 * np.pi * r
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    pts = np.c_[x, y] + np.random.randn(n, 2) * noise
    return pts

def make_hollow_square(n=400, noise=0.01):
    side = n // 4
    xs = np.linspace(-1, 1, side)
    top = np.c_[xs, np.ones_like(xs)]
    bottom = np.c_[xs, -np.ones_like(xs)]
    left = np.c_[np.ones_like(xs), xs]
    right = np.c_[ -np.ones_like(xs), xs]
    pts = np.vstack([top, bottom, left, right]) + np.random.randn(n, 2) * noise
    return pts

SHAPES = dict(half_moons=make_half_moons,
              zigzag=make_zigzag,
              spiral=make_spiral,
              hollow_square=make_hollow_square)

# ------------------------------------------------------------
# 2.  Unnormalized base-10 float tokeniser  (B=10,  E=1 exp digit,  M=5 mantissa digits)
# ------------------------------------------------------------
class FloatTokenizer:
    def __init__(self, base=10, E=1, M=5):
        self.base, self.E, self.M = base, E, M
        # Vocab:  10 digits 0-9  + sign tokens  + exponent-sign tokens
        self.vocab = ['<pad>','<s>','</s>','<nan>','+','-'] + [str(i) for i in range(base)]
        self.pad, self.bos, self.eos = 0, 1, 2
        self.nan = 3; self.sgn = {'+':4,'-':5}; self.d0 = 6

    def _digit(self, d): return self.d0 + d   # 0-9 → 6-15

    def encode(self, y: float):
        if math.isnan(y): return [self.nan]
        sign = '+' if y >= 0 else '-'
        y = abs(y) + 1e-12
        exponent = int(math.floor(math.log(y, self.base)))
        exponent = exponent + 1
        mant_full = y / (self.base**exponent)
        mant = mant_full - int(mant_full)
        exp_sign = '+' if exponent >= 0 else '-'
        exponent = abs(exponent)
        exp_digits = [self._digit((exponent // (self.base**i)) % self.base)
                      for i in range(self.E-1,-1,-1)]
        mantissa_digits = []
        for _ in range(self.M):
            mant *= self.base
            d = int(mant)
            mant -= d
            mantissa_digits.append(self._digit(d))
        tokens = [self.sgn[sign], self.sgn[exp_sign]] + exp_digits + mantissa_digits
        return [self.bos] + tokens + [self.eos]

    def decode(self, tokens):
        if tokens[0] == self.nan: return float('nan')
        idx = 1  # skip <s>
        sign = 1 if tokens[idx]==self.sgn['+'] else -1; idx+=1
        exp_sign = 1 if tokens[idx]==self.sgn['+'] else -1; idx+=1
        exponent = 0
        for t in tokens[idx:idx+self.E]:
            exponent = exponent*self.base + (t-self.d0); idx+=1
        exponent *= exp_sign
        mant = 0
        for t in tokens[idx:idx+self.M]:
            mant = mant*self.base + (t-self.d0)
        mant /= self.base**self.M
        return sign * (self.base**exponent) * (1+mant)

    @property
    def vocab_size(self): return len(self.vocab)

# ------------------------------------------------------------
# 3.  Dataset & DataLoader
# ------------------------------------------------------------
class ShapeDataset(Dataset):
    def __init__(self, pts, tokenizer):
        self.x = torch.tensor(pts[:,0:1], dtype=torch.float32)
        self.y = pts[:,1].astype(float)
        self.tok = tokenizer
        self.y_tok = [torch.tensor(self.tok.encode(v), dtype=torch.long) for v in self.y]

    def __len__(self):  return len(self.x)
    def __getitem__(self, i): return self.x[i], self.y_tok[i]

def collate(batch, pad=0):
    xs, ys = zip(*batch)
    xs = torch.stack(xs)
    L = max(len(y) for y in ys)
    y_pad = torch.full((len(ys), L), pad, dtype=torch.long)
    for i,y in enumerate(ys): y_pad[i,:len(y)] = y
    return xs, y_pad

# ------------------------------------------------------------
# 4.  Encoder + Transformer Decoder model
# ------------------------------------------------------------
class DecoderRegressor(nn.Module):
    def __init__(self, vocab, d_model=32, n_layers=1, n_heads=1):
        super().__init__()
        self.d_model = d_model
        self.embed_tok = nn.Embedding(vocab, d_model, padding_idx=0)
        self.embed_pos = nn.Embedding(64, d_model)                 # enough for len<=64
        self.encoder = nn.Sequential(nn.Linear(1, d_model), nn.ReLU(),
                                     nn.Linear(d_model, d_model))
        self.tdecoder = nn.TransformerDecoder(
              nn.TransformerDecoderLayer(d_model, n_heads, dim_feedforward=4*d_model),
              num_layers=n_layers)
        self.output = nn.Linear(d_model, vocab)

    def forward(self, x, y_tok):
        # x: (B,1) , y_tok: (B,L)
        B,L = y_tok.shape
        mem = self.encoder(x).unsqueeze(0)              # (1,B,d_model)
        tok_emb = self.embed_tok(y_tok) + self.embed_pos(torch.arange(L,device=x.device))
        tgt = tok_emb.transpose(0,1)                    # (L,B,d_model)
        mask = nn.Transformer.generate_square_subsequent_mask(L).to(x.device)
        out = self.tdecoder(tgt, mem, tgt_mask=mask).transpose(0,1)    # (B,L,d_model)
        return self.output(out)                         # (B,L,V)

    def sample(self, x, tok, temp=1.0, max_len=20):
        self.eval()
        with torch.no_grad():
            mem = self.encoder(x.unsqueeze(0)).unsqueeze(0)
            seq = torch.tensor([tok.bos], device=x.device).unsqueeze(0)   # (1,1)
            for pos in range(max_len-1):
                logits = self.forward(x.unsqueeze(0), seq)[:,-1]/temp
                probs = F.softmax(logits, -1)
                next_tok = torch.multinomial(probs, 1)
                seq = torch.cat([seq, next_tok], dim=1)
                if next_tok.item() == tok.eos: break
        return tok.decode(seq.squeeze().tolist())

# ------------------------------------------------------------
# 5.  Training
# ------------------------------------------------------------
def train(shape='half_moons', epochs=300, bs=64, device='cuda'):
    tok = FloatTokenizer()
    pts = SHAPES[shape]()
    ds = ShapeDataset(pts, tok)
    dl = DataLoader(ds, batch_size=bs, shuffle=True, collate_fn=collate)
    model = DecoderRegressor(tok.vocab_size).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    for epoch in range(epochs):
        for x,y in dl:
            x,y = x.to(device), y.to(device)
            logits = model(x, y[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, tok.vocab_size),
                                   y[:,1:].reshape(-1), ignore_index=tok.pad)
            opt.zero_grad(); loss.backward(); opt.step()
        if epoch%50==0: print(f'{epoch:>3}  loss={loss.item():.4f}')
    return model, tok

# ------------------------------------------------------------
# 6.  Visualisation   (Fig. 8 style)
# ------------------------------------------------------------
def plot_density(model, tok, shape='half_moons', N=400, grid=200):
    pts = SHAPES[shape](N, noise=0)  # ground-truth clean samples
    x_min, x_max = pts[:,0].min()-0.2, pts[:,0].max()+0.2
    y_min, y_max = pts[:,1].min(), pts[:,1].max()

    xs = torch.linspace(x_min, x_max, grid)
    preds = [model.sample(torch.tensor([x]), tok, temp=1.0) for x in xs]

    preds = np.array(preds)
    outliers = (preds < y_min) | (preds > y_max)
    outlier_fraction = outliers.mean()

    # Clip preds to stay within y-range for plotting
    preds_clipped = np.clip(preds, y_min, y_max)

    print(f'Fraction of predictions outside ground-truth y-range: {outlier_fraction:.4f}')

    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1); plt.title('Ground truth'); plt.scatter(pts[:,0], pts[:,1], s=5)
    plt.subplot(1,2,2); plt.title('Decoder samples')
    plt.scatter(xs.numpy(), preds_clipped, s=5, alpha=0.6)
    plt.tight_layout(); plt.show()

if __name__ == '__main__':
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    model, tok = train('half_moons', epochs=300, device='cpu')
    plot_density(model, tok, 'spiral')