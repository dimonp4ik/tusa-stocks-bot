"""Can the features we already collect predict a trade's outcome?

Three days of work tuned thresholds one feature at a time. This asks the
mathematical version of the question: let a model weight ALL of them jointly
and see whether anything is predictable out of sample.

Strictly honest setup:
  * only fields known AT ENTRY — no mae_r, no exit_*, nothing outcome-derived;
  * fit on the FIRST half of the window, score on the SECOND (never a random
    split — trades are ordered in time and a random split leaks the regime);
  * AUC on the held-out half is the verdict. 0.50 is a coin flip.

A previous attempt on 2026-08-07 found AUC 0.52 for price-shaped features, but
that ran on the pre-fix fill model and before extension_atr / room_atr /
vol_atr_pct existed, so it is worth redoing rather than citing.

Usage: python tools_predictive.py bt_book.csv
"""
import csv, sys
import numpy as np

NUM = ["mtf_score", "volume_ratio", "rsi", "eff_ratio", "vol_atr_pct",
       "extension_atr", "room_atr", "entry_range_atr", "quality_score",
       "trend_score", "volatility_score", "entry_quality_score",
       "portfolio_risk_score", "knn_score", "bos_candles_ago", "zone_age_bars",
       "risk_mult", "premium", "sniper", "tp1_beyond_level"]
CAT = ["direction", "session", "trend_1h", "trend_4h", "entry_source", "swing_trend"]


def load(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    for r in rows:
        r["_t"] = float(r["entry_time"])
        r["_y"] = 1.0 if float(r["net_r"]) > 0 else 0.0
    rows.sort(key=lambda x: x["_t"])
    return rows


def featurize(rows):
    cats = {c: sorted({(r.get(c) or "") for r in rows}) for c in CAT}
    names, cols = [], []
    for f in NUM:
        v = []
        ok = True
        for r in rows:
            try:
                v.append(float(r.get(f) or 0.0))
            except (TypeError, ValueError):
                ok = False
                break
        if ok and len(set(v)) > 1:
            names.append(f)
            cols.append(v)
    for c, vals in cats.items():
        for val in vals[:-1]:           # drop one level to avoid collinearity
            names.append(f"{c}={val}")
            cols.append([1.0 if (r.get(c) or "") == val else 0.0 for r in rows])
    X = np.array(cols, dtype=float).T
    y = np.array([r["_y"] for r in rows], dtype=float)
    return X, y, names


def fit_logit(X, y, l2=1.0, iters=4000, lr=0.08):
    Xb = np.hstack([np.ones((len(X), 1)), X])
    w = np.zeros(Xb.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Xb @ w, -30, 30)))
        g = Xb.T @ (p - y) / len(y)
        g[1:] += l2 * w[1:] / len(y)
        w -= lr * g
    return w


def predict(w, X):
    Xb = np.hstack([np.ones((len(X), 1)), X])
    return 1.0 / (1.0 + np.exp(-np.clip(Xb @ w, -30, 30)))


def auc(y, p):
    order = np.argsort(p)
    ranks = np.empty(len(p), dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    pos, neg = y.sum(), (1 - y).sum()
    if pos == 0 or neg == 0:
        return 0.5
    return (ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "bt_now.csv"
    rows = load(path)
    X, y, names = featurize(rows)
    n = len(rows)
    half = n // 2
    mu, sd = X[:half].mean(0), X[:half].std(0)
    sd[sd == 0] = 1.0
    Xs = (X - mu) / sd
    print(f"сделок {n}, признаков {len(names)}, доля побед {y.mean():.3f}")
    print(f"обучение на первых {half}, проверка на последних {n - half}\n")

    w = fit_logit(Xs[:half], y[:half])
    p_in = predict(w, Xs[:half])
    p_out = predict(w, Xs[half:])
    a_in, a_out = auc(y[:half], p_in), auc(y[half:], p_out)
    print(f"AUC на обучении  {a_in:.4f}")
    print(f"AUC вне выборки  {a_out:.4f}   {'ЕСТЬ СИГНАЛ' if a_out > 0.55 else 'шум' if a_out < 0.53 else 'на грани'}")

    # обратная проверка: учим на второй половине, меряем на первой
    mu2, sd2 = X[half:].mean(0), X[half:].std(0)
    sd2[sd2 == 0] = 1.0
    Xs2 = (X - mu2) / sd2
    w2 = fit_logit(Xs2[half:], y[half:])
    a_rev = auc(y[:half], predict(w2, Xs2[:half]))
    print(f"AUC в обратную сторону {a_rev:.4f}")

    print("\nвес признака (по обучающей половине, стандартизовано):")
    idx = np.argsort(-np.abs(w[1:]))
    for i in idx[:12]:
        print(f"   {names[i]:<26}{w[i+1]:>+8.4f}")

    # что даёт отбор по предсказанной вероятности на held-out половине
    r_out = np.array([float(r["net_r"]) for r in rows[half:]])
    print("\nотбор по предсказанию на второй половине:")
    base = r_out.mean()
    print(f"   все сделки          {len(r_out):>4}  {base:>+7.4f}R")
    for q in (0.25, 0.5):
        thr = np.quantile(p_out, q)
        keep = p_out >= thr
        print(f"   верхние {100*(1-q):>2.0f}%        {keep.sum():>4}  {r_out[keep].mean():>+7.4f}R"
              f"   отсечённые {r_out[~keep].mean():>+7.4f}R")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
