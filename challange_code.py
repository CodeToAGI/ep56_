"""EP56 - Model Compression: prune, quantize and distil a digits classifier, then measure size, speed and accuracy.
    python ep56_compression.py     # 5 seeds, a few minutes on a CPU; writes ep56_results.json
pip install torch scikit-learn numpy
"""
import copy, io, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from sklearn.datasets import load_digits
# ── 1. CONFIG ─────────────────────────────────────────────────────────────
SEEDS = [0, 1, 2, 3, 4]
EPOCHS = 60                          # training epochs for every model
FT = 10                              # extra epochs to recover after pruning or fake-quantizing
BIG = 512                            # hidden width of the big model
SMALL = 16                           # hidden width of the small student
SPARSITY = [0.0, 0.5, 0.8, 0.9, 0.95, 0.98]     # share of weights removed by pruning
BITS = [8, 4, 3, 2]                  # weight bit-widths for quantization
TEMP = 4.0                           # distillation: temperature that softens the teacher
ALPHA = 0.5                          # distillation: weight of the teacher in the loss
# ── 2. DATA: 8x8 digits, a fixed 1,437 / 360 split ────────────────────────
def load():
    x, y = load_digits(return_X_y=True)
    x, y = torch.tensor(x / 16.0, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
    perm = torch.tensor(np.random.RandomState(0).permutation(len(x)), dtype=torch.long); cut = int(0.8 * len(x))
    return x[perm[:cut]], y[perm[:cut]], x[perm[cut:]], y[perm[cut:]]
# ── 3. MODEL: three linear layers; bits=k fakes k-bit weights in the forward pass ──
def fq(w, bits):                     # symmetric rounding to 2^bits levels; the gradient skips the round (STE)
    qmax = 2 ** (bits - 1) - 1
    s = w.detach().abs().max().clamp_min(1e-8) / qmax
    return w + ((w / s).round().clamp(-qmax, qmax) * s - w).detach()
class Net(nn.Module):
    def __init__(self, h, bits=None):
        super().__init__()
        self.fc = nn.ModuleList([nn.Linear(64, h), nn.Linear(h, h), nn.Linear(h, 10)])
        self.bits = bits
    def forward(self, x):
        for i, l in enumerate(self.fc):
            x = F.linear(x, fq(l.weight, self.bits) if self.bits else l.weight, l.bias)
            x = F.relu(x) if i < 2 else x
        return x
# ── 4. TRAIN: plain, distilled (teacher=...) or pruned (masks=...) ────────
def fit(model, x, y, epochs=EPOCHS, teacher=None, masks=None):
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(epochs):
        order = torch.randperm(len(x))
        for i in range(0, len(x), 64):
            b = order[i:i + 64]; out = model(x[b]); loss = F.cross_entropy(out, y[b])
            if teacher is not None:
                with torch.no_grad(): soft = F.softmax(teacher(x[b]) / TEMP, 1)
                kd = F.kl_div(F.log_softmax(out / TEMP, 1), soft, reduction="batchmean")
                loss = (1 - ALPHA) * loss + ALPHA * TEMP ** 2 * kd
            opt.zero_grad(); loss.backward(); opt.step()
            if masks is not None:
                with torch.no_grad(): [l.weight.mul_(m) for l, m in zip(model.fc, masks)]
    return model
# ── 5. THE TECHNIQUES: prune, convert to int8, measure ────────────────────
def prune(model, sparsity):          # global magnitude pruning: zero the smallest weights of all layers
    w = torch.cat([l.weight.detach().abs().flatten() for l in model.fc]); k = int(sparsity * len(w))
    cut = w.kthvalue(k).values if k > 0 else -1.0
    masks = [(l.weight.detach().abs() > cut).float() for l in model.fc]
    with torch.no_grad(): [l.weight.mul_(m) for l, m in zip(model.fc, masks)]
    return masks
def plain(net):                      # the same layers as an nn.Sequential, so PyTorch can swap in int8 ones
    return nn.Sequential(net.fc[0], nn.ReLU(), net.fc[1], nn.ReLU(), net.fc[2]).eval()
def int8(net):
    return torch.ao.quantization.quantize_dynamic(plain(net), {nn.Linear}, dtype=torch.qint8)
def correct(model, x, y):
    with torch.no_grad(): return int((model(x).argmax(1) == y).sum())
def kbytes(model):                   # size of the saved weights file
    buf = io.BytesIO(); torch.save(model.state_dict(), buf); return buf.getbuffer().nbytes / 1024
def millis(model, x, reps=200):      # median time to classify the whole test set once
    with torch.no_grad():
        for _ in range(20): model(x)
        t = []
        for _ in range(reps): a = time.perf_counter(); model(x); t.append(time.perf_counter() - a)
    return float(np.median(t)) * 1000
# ── 6. RUN: every seed, every technique, one results file ─────────────────
def run():
    torch.set_num_threads(1)
    xtr, ytr, xte, yte = load(); R = {}
    def put(key, v): R.setdefault(key, []).append(v)
    for s in SEEDS:
        torch.manual_seed(s)
        big = fit(Net(BIG), xtr, ytr); put("float", correct(big, xte, yte))
        for sp in SPARSITY:
            m = copy.deepcopy(big); masks = prune(m, sp); put(f"prune_{sp}", correct(m, xte, yte))
            put(f"nnz_{sp}", sum(int(torch.count_nonzero(l.weight)) for l in m.fc) / sum(l.weight.numel() for l in m.fc))
            put(f"tuned_{sp}", correct(fit(m, xtr, ytr, FT, masks=masks), xte, yte))
        put("pruned_kb", kbytes(m))  # a pruned model saved as a dense file
        for b in BITS:
            m = copy.deepcopy(big); m.bits = b; put(f"ptq_{b}", correct(m, xte, yte))
            put(f"qat_{b}", correct(fit(m, xtr, ytr, FT), xte, yte))
        q = int8(big); put("int8", correct(q, xte, yte))
        put("float_kb", kbytes(plain(big))); put("int8_kb", kbytes(q))
        put("float_ms", millis(plain(big), xte)); put("int8_ms", millis(q, xte))
        torch.manual_seed(100 + s); put("alone", correct(fit(Net(SMALL), xtr, ytr), xte, yte))
        torch.manual_seed(100 + s); put("kd", correct(fit(Net(SMALL), xtr, ytr, teacher=big), xte, yte))
        print(f"seed {s}: float {R['float'][-1]} int8 {R['int8'][-1]} student {R['alone'][-1]} -> {R['kd'][-1]} (of {len(yte)})")
    R.update(test_n=len(yte), seeds=SEEDS, sparsity=SPARSITY, bits=BITS,
             big_params=sum(p.numel() for p in Net(BIG).parameters()), small_params=sum(p.numel() for p in Net(SMALL).parameters()))
    Path(__file__).with_name("ep56_results.json").write_text(json.dumps(R))
    return R

if __name__ == "__main__":
    R = run(); pc = lambda k: 100 * np.mean(R[k]) / R["test_n"]; mu = lambda k: float(np.mean(R[k]))
    print(f"float32 {pc('float'):.2f}% {mu('float_kb'):.0f} KB {mu('float_ms'):.2f} ms | int8 {pc('int8'):.2f}% {mu('int8_kb'):.0f} KB {mu('int8_ms'):.2f} ms")
    for sp in SPARSITY: print(f"prune {sp:.0%}: no retraining {pc(f'prune_{sp}'):.1f}% | fine-tuned {pc(f'tuned_{sp}'):.1f}%")
    for b in BITS: print(f"{b}-bit weights: post-training {pc(f'ptq_{b}'):.1f}% | quantization-aware {pc(f'qat_{b}'):.1f}%")
    print(f"student {R['small_params']:,} params: alone {pc('alone'):.2f}% | distilled {pc('kd'):.2f}% | teacher {R['big_params']:,} params {pc('float'):.2f}%")
