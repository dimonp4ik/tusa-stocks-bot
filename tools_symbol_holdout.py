"""Validate a sizing rule on symbols it was never fitted on.

The stocks bot has no historical regimes — OKX listed those perps in Feb-Mar
2026 — so every rule there rests on one window cut into thirds, which is a
weak test: the three stretches share a market. Symbols do not. Splitting the
book by TICKER gives a genuinely independent hold-out that needs no history at
all, and it answers a different question from a time split: not "does this
survive a change of regime" but "is this a property of the strategy or of a
handful of instruments".

Both matter. A rule that only works on the tickers it was derived from is
curve-fitting no matter how many time slices agree.

Usage: python tools_symbol_holdout.py <book.csv> <rule>
  rules: orderly | open | chop | london | overlap | room
"""
import csv, sys


def num(r, k):
    try:
        return float(r.get(k) or 0.0)
    except (TypeError, ValueError):
        return 0.0


RULES = {
    "orderly": lambda r: (num(r, "eff_ratio") >= 0.31
                          and num(r, "vol_atr_pct") < 0.0044
                          and num(r, "bos_extension_atr") >= 1.271),
    "open":    lambda r: ((r.get("session") or "") == "OPEN"
                          and num(r, "volume_ratio") >= 2.5),
    "chop":    lambda r: (num(r, "volume_ratio") >= 2.11
                          and num(r, "eff_ratio") < 0.4476
                          and num(r, "vol_atr_pct") >= 0.006268),
    "london":  lambda r: ((r.get("session") or "") == "LONDON"
                          and num(r, "volume_ratio") >= 2.0),
    "overlap": lambda r: ((r.get("session") or "") == "OVERLAP"
                          and num(r, "volume_ratio") < 2.5),
    "room":    lambda r: num(r, "room_atr") >= 3.5,
    # Stocks desk, shipped 2026-08-29. Ported here because this tool was written
    # for exactly this bot's problem (no historical regimes) and had never been
    # run on it.
    "volspike": lambda r: num(r, "volume_ratio") >= 4.0,
    # The win definition below counts any TRAIL as a win, which overstates by
    # ~4pp — the trail is floored at breakeven in PRICE, so a flat exit goes
    # negative after fees. Left as-is so figures stay comparable with the crypto
    # runs; read rr(), not wr(), when the two disagree.
}


def wr(g):
    return sum(1 for r in g if r["outcome"] in ("TP1", "TP2", "TRAIL")) / max(1, len(g))


def rr(g):
    return sum(float(r["net_r"] or 0) for r in g) / max(1, len(g))


# A half needs enough trades IN THE SUBSET for its estimate to mean anything.
# Learned by using this tool on the open-space rule, whose subset is 98 trades
# spread over 15 symbols — 3 to 9 each, with per-symbol differences running
# from -1.35 to +1.10. Splitting that in half just deals the outliers unevenly
# and prints "sign disagreed", which reads like an adverse result and is really
# no result at all. An underpowered test is not a negative test.
MIN_SUB = 60


def report(rows, hit, label):
    sub = [r for r in rows if hit(r)]
    rest = [r for r in rows if not hit(r)]
    if len(sub) < MIN_SUB or len(rest) < MIN_SUB:
        print(f"  {label:12s} НЕДОСТАТОЧНО МОЩНОСТИ: {len(sub)} сд в подмножестве "
              f"(нужно {MIN_SUB}) — тест не применим, это НЕ отрицательный ответ")
        return None
    lift = rr(sub) - rr(rest)
    print(f"  {label:12s} подмн. {len(sub):4d}сд {wr(sub)*100:5.1f}% {rr(sub):+.3f}"
          f"  |  прочее {len(rest):4d}сд {wr(rest)*100:5.1f}% {rr(rest):+.3f}"
          f"  |  прирост {lift:+.3f}")
    return lift


def main():
    args = [a for a in sys.argv[1:]]
    name = args[-1] if args[-1] in RULES else "chop"
    paths = [a for a in args if a not in RULES]
    hit = RULES[name]
    # Pool the books. The regime question is answered separately by running
    # each window end-to-end; what is asked HERE is whether the rule is a
    # property of the strategy or of particular instruments, and for that,
    # pooling is the right move — it is the only way the per-half subsets get
    # large enough to mean anything.
    rows = []
    for p in paths:
        rows += list(csv.DictReader(open(p, encoding="utf-8")))
    syms = sorted({r["symbol"] for r in rows})
    # Deterministic split, alternating by sorted name — not random, so the
    # result is reproducible, and not by volume or trade count, which would
    # correlate with the very thing being tested.
    a = {s for i, s in enumerate(syms) if i % 2 == 0}
    b = set(syms) - a
    print(f"{len(paths)} книг(и), {len(rows)} сделок  правило: {name}")
    print(f"тикеров: {len(syms)} -> {len(a)} / {len(b)}")
    la = report([r for r in rows if r["symbol"] in a], hit, "половина A")
    lb = report([r for r in rows if r["symbol"] in b], hit, "половина B")
    print()
    if la is None or lb is None:
        print("  вердикт: недостаточно данных")
    elif (la > 0) == (lb > 0):
        print(f"  вердикт: знак СОВПАЛ на обеих половинах ({la:+.3f} / {lb:+.3f}) — "
              f"свойство стратегии, а не отдельных тикеров")
    else:
        print(f"  вердикт: знак РАЗОШЁЛСЯ ({la:+.3f} / {lb:+.3f}) — "
              f"держится на части инструментов")


if __name__ == "__main__":
    main()
