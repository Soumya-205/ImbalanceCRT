import numpy as np

def crc_threshold_fnr(pos_scores, alpha):
    """Post-hoc Conformal Risk Control for FNR.
    pos_scores: model scores for the POSITIVE calibration examples only.
    Loss per positive: 1[score < lam]; bound B = 1.
    Returns the largest lam with (1 + #missed) / (N + 1) <= alpha.
    Predict positive iff score >= lam. If infeasible, returns (0.0, False)."""
    s = np.sort(np.asarray(pos_scores, dtype=float))
    k = int(np.floor(alpha * (len(s) + 1) - 1))
    if k < 0:
        return 0.0, False
    return float(s[k]), True

def fnr_fpr(y_true, scores, lam):
    y_true, scores = np.asarray(y_true), np.asarray(scores)
    pred = scores >= lam
    pos = y_true == 1
    return float(np.mean(~pred[pos])), float(np.mean(pred[~pos]))