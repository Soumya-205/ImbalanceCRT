import numpy as np
import torch
import torch.nn as nn
from models import MLP, predict_scores
from crc import crc_threshold_fnr, fnr_fpr
from data import get_imbalance_sweep_splits
from config import TARGET_FNR_LEVELS

def to_np(X, y):
    return X.values.astype(np.float32), y.values.astype(np.float32)

def train_bce(X, y, class_weighted, seed, epochs=30, lr=1e-3, batch_size=256):
    torch.manual_seed(seed)
    model = MLP(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    pw = None
    if class_weighted:
        n_pos = float(y.sum())
        pw = torch.tensor([(len(y) - n_pos) / n_pos], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    for _ in range(epochs):
        perm = torch.randperm(len(yt))
        for i in range(0, len(yt), batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss_fn(model(Xt[idx]), yt[idx]).backward()
            opt.step()
    return model

def run_baseline(train, cal, test, class_weighted, seed):
    X_tr, y_tr = to_np(*train)
    X_cal, y_cal = to_np(*cal)
    X_te, y_te = to_np(*test)
    model = train_bce(X_tr, y_tr, class_weighted, seed)
    s_cal, s_te = predict_scores(model, X_cal), predict_scores(model, X_te)
    out = []
    for alpha in TARGET_FNR_LEVELS:
        lam, ok = crc_threshold_fnr(s_cal[y_cal == 1], alpha)
        fnr, fpr = fnr_fpr(y_te, s_te, lam)
        out.append(dict(alpha=alpha, fnr=fnr, fpr=fpr, lam=lam, feasible=ok))
    return out

if __name__ == "__main__":
    seed = 0
    train, cal, test = get_imbalance_sweep_splits(seed)[0.05]
    for name, cw in [("A (BCE)", False), ("B (weighted BCE)", True)]:
        for r in run_baseline(train, cal, test, cw, seed):
            print(f"Model {name} | alpha={r['alpha']:.2f} | "
                  f"FNR={r['fnr']:.3f} | FPR={r['fpr']:.3f} | lam={r['lam']:.4f}")