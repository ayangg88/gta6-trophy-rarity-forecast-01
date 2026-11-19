"""gta6-trophy-rarity-forecast-01: forecast GTA VI trophy earn rates on Exophase.

For every trophy, forecast its earn rate at the case's target as five
quantiles (5%, 25%, 50%, 75%, 95%), scored by pinball loss on the logit
scale. Python 3.13, standard library only, deterministic, no network.

Model
-----
A rate is a ratio of two growing quantities, so each trophy's rate is
decomposed into earners over tracked players: E = rate / 100 * N with
N = players_tracked. log(E) and log(N) are each fitted against
x = ln(1 + hours since release, 2026-11-19T00:00:00Z) by recency-weighted
least squares (24 h half-life), extrapolated to the target, and the median
rate is 100 * E / N. Each trophy's earner slope is shrunk toward its
tier's median slope, weight n / (n + 4) on its own slope, so noisy or
rare trophies stay sane; the platinum trophy is shrunk heavily toward
a slow-growth prior and its median is capped at single digits.

Flat history (every read identical, as on launch night): no trend can be
estimated. The median is the last rate shifted down for dilution by new
players -- last * 0.3 while the tracked population is still small
(under 10,000), else last / sqrt(1 + horizon_days) -- with a wide
spread, because Exophase's first refresh (638 -> 3,868 tracked) cut
every rate to roughly a fifth. A trophy at 100% in the last read stays
at 100%.

Spread (logit units): flat history, sigma = max(0.5, 1.2 * sqrt(days)),
capped at 2.5; otherwise sigma combines the random-walk term
(0.5 * sqrt(days)), the fit residual and the slope uncertainty times the
distance to the target in x, floored at 0.35 and capped at 2.5.
Quantiles: q = 100 / (1 + exp(-(logit(median) + z * sigma))), rounded
to 4 decimals half-up, enforced non-decreasing.

A case that fails anywhere still writes a valid output: the last read's
rates with a wide spread, so no case is ever invalid.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)
HALF_LIFE_HOURS = 24.0
SHRINK_PRIOR = 4.0
PLATINUM_CAP = 5.0
SIGMA_FLOOR = 0.35
SIGMA_CAP = 2.5


def parse_time(text):
    """Parse an ISO 8601 timestamp (trailing Z allowed) to an aware datetime."""
    return datetime.fromisoformat(str(text).replace("Z", "+00:00"))


def hours_since_release(moment, release):
    return max(0.0, (moment - release).total_seconds() / 3600.0)


def clamp_rate(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    """x rounded to 4 decimals, half up (as the starter rounds it).

    A whole number comes back as an int, so json.dumps writes 100
    where 100.0 would also be valid but longer.
    """
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def median_of(values):
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    if n % 2:
        return ordered[n // 2]
    return 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])


def wls(xs, ys, ws):
    """Recency-weighted least squares of y on x.

    Returns (intercept, slope, residual_std, slope_se, n). With fewer
    than 3 distinct x values the slope is 0 and the intercept is the
    weighted mean, so the fit is flat by construction.
    """
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.5, 0
    sw = sum(ws)
    xm = sum(w * x for w, x in zip(ws, xs)) / sw
    ym = sum(w * y for w, y in zip(ws, ys)) / sw
    distinct = len({round(x, 6) for x in xs})
    sxx = sum(w * (x - xm) ** 2 for w, x in zip(ws, xs))
    if distinct < 3 or sxx <= 1e-12:
        resid = [y - ym for y in ys]
        var = sum(w * r * r for w, r in zip(ws, resid)) / sw if n else 0.0
        return ym, 0.0, math.sqrt(max(var, 0.0)), 0.5, n
    sxy = sum(w * (x - xm) * (y - ym) for w, x, y in zip(ws, xs, ys))
    slope = sxy / sxx
    intercept = ym - slope * xm
    resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    var = sum(w * r * r for w, r in zip(ws, resid)) / sw
    if n > 2:
        var *= n / (n - 2)
    se = math.sqrt(max(var, 0.0) / sxx) if sxx > 0 else 0.5
    return intercept, slope, math.sqrt(max(var, 0.0)), se, n


def quantiles(median, sigma):
    """Five quantiles around a median rate, rounded half-up, non-decreasing."""
    base = logit(median)
    out = []
    prev = 0.0
    for z in Z:
        q = round4(inv_logit(base + z * sigma))
        q = min(100, max(0, q))
        if q < prev:
            q = prev
        out.append(q)
        prev = q
    return out


def fallback_rows(case):
    """Last read's rates with a wide spread; always a valid output."""
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    last_rates = history[-1].get("rates") if history else None
    rows = []
    for i, trophy in enumerate(trophies):
        rate = None
        if last_rates is not None and i < len(last_rates):
            rate = last_rates[i]
        if rate is None:
            rate = 1.0
        if rate >= 100:
            rows.append({"id": trophy["id"], "q": [99.9, 100, 100, 100, 100]})
        else:
            rows.append({"id": trophy["id"], "q": quantiles(clamp_rate(rate), 2.0)})
    return {"trophies": rows}


def forecast_case(case):
    trophies = case["trophies"]
    history = case["history"]
    if not history:
        raise ValueError("case has no history")
    horizon_days = case["horizon_minutes"] / 1440.0
    target = parse_time(case["target"])
    release_text = (case.get("game") or {}).get("release_date") or "2026-11-19"
    try:
        release = datetime.fromisoformat(release_text).replace(tzinfo=timezone.utc)
    except ValueError:
        release = RELEASE
    x_target = math.log1p(hours_since_release(target, release))

    # One (x, weight) per read, plus the tracked population per read.
    read_times = [parse_time(h["at"]) for h in history]
    t_last = read_times[-1]
    xs = [math.log1p(hours_since_release(t, release)) for t in read_times]
    weights = [
        0.5 ** ((t_last - t).total_seconds() / 3600.0 / HALF_LIFE_HOURS)
        for t in read_times
    ]
    populations = []
    for h in history:
        n = h.get("players_tracked")
        populations.append(float(n) if isinstance(n, (int, float)) and n > 0 else None)

    # Flat history: every rate and the tracked count unchanged throughout.
    flat = all(p == populations[0] for p in populations)
    if flat:
        for i in range(len(trophies)):
            seen = {h["rates"][i] for h in history if h["rates"][i] is not None}
            if len(seen) > 1:
                flat = False
                break

    last = history[-1]
    last_rates = last["rates"]
    last_n = populations[-1] or 0.0

    # Tier medians of the last read's positive rates, for the zero-rate floor.
    tier_last = {}
    for trophy, rate in zip(trophies, last_rates):
        if rate is not None and rate > 0:
            tier_last.setdefault(trophy["tier"], []).append(rate)
    tier_floor = {
        tier: max(0.01, 0.1 * median_of(rates)) for tier, rates in tier_last.items()
    }

    if flat:
        sigma = min(SIGMA_CAP, max(0.5, 1.2 * math.sqrt(horizon_days)))
        rows = []
        for i, trophy in enumerate(trophies):
            rate = last_rates[i]
            if rate is None:
                rate = tier_floor.get(trophy["tier"], 1.0)
            if rate >= 100:
                rows.append({"id": trophy["id"], "q": [99.9, 100, 100, 100, 100]})
                continue
            if rate <= 0:
                med = tier_floor.get(trophy["tier"], 0.1)
            elif last_n < 10000:
                med = rate * 0.3
            else:
                med = rate / math.sqrt(1.0 + horizon_days)
            rows.append({"id": trophy["id"], "q": quantiles(clamp_rate(med), sigma)})
        return {"trophies": rows}

    # Population fit: log(N) on x, shared by every trophy.
    n_xs, n_ys, n_ws = [], [], []
    for x, w, pop in zip(xs, weights, populations):
        if pop is not None:
            n_xs.append(x)
            n_ys.append(math.log(pop))
            n_ws.append(w)
    b_n, s_n, r_n, se_n, _ = wls(n_xs, n_ys, n_ws)

    # Per-trophy earner fits: log(E) on x, E = rate / 100 * N.
    fits = []
    for i, trophy in enumerate(trophies):
        e_xs, e_ys, e_ws = [], [], []
        for j, h in enumerate(history):
            rate = h["rates"][i]
            pop = populations[j]
            if rate is None or pop is None or rate <= 0:
                continue
            e = rate / 100.0 * pop
            if e <= 0:
                continue
            e_xs.append(xs[j])
            e_ys.append(math.log(e))
            e_ws.append(weights[j])
        fits.append(wls(e_xs, e_ys, e_ws))

    # Tier median earner slopes, for shrinkage.
    tier_slopes = {}
    for trophy, fit in zip(trophies, fits):
        if fit[4] >= 3:
            tier_slopes.setdefault(trophy["tier"], []).append(fit[1])
    all_slopes = [s for slopes in tier_slopes.values() for s in slopes]
    global_slope = median_of(all_slopes) if all_slopes else 0.0
    tier_median = {
        tier: median_of(slopes) for tier, slopes in tier_slopes.items()
    }

    rows = []
    for i, trophy in enumerate(trophies):
        rate = last_rates[i]
        if rate is None:
            rate = tier_floor.get(trophy["tier"], 1.0)
        if rate >= 100:
            rows.append({"id": trophy["id"], "q": [99.9, 100, 100, 100, 100]})
            continue
        b_e, s_e, r_e, se_e, n_e = fits[i]
        if rate <= 0:
            med = tier_floor.get(trophy["tier"], 0.1)
            sigma = min(SIGMA_CAP, max(SIGMA_FLOOR, 0.5 * math.sqrt(horizon_days)))
            rows.append({"id": trophy["id"], "q": quantiles(clamp_rate(med), sigma)})
            continue
        # Shrink the earner slope toward the tier median slope.
        prior = tier_median.get(trophy["tier"], global_slope)
        w = n_e / (n_e + SHRINK_PRIOR) if n_e else 0.0
        slope = w * s_e + (1.0 - w) * prior
        slope = max(-1.5, min(1.5, slope))
        if trophy["tier"] == "platinum":
            # The platinum grows only as players finish everything:
            # shrink heavily toward a slow-growth prior and cap low.
            slope = 0.25 * slope
        log_e = b_e + slope * x_target
        log_n = b_n + s_n * x_target
        med = 100.0 * math.exp(log_e - log_n)
        if trophy["tier"] == "platinum":
            med = min(med, PLATINUM_CAP)
        med = clamp_rate(med)
        resid = math.sqrt(r_e * r_e + r_n * r_n)
        unc = math.sqrt(se_e * se_e + se_n * se_n)
        dist = max(0.0, x_target - xs[-1])
        sigma = math.sqrt(
            (0.5 * math.sqrt(horizon_days)) ** 2
            + resid * resid
            + (unc * dist) ** 2
        )
        sigma = min(SIGMA_CAP, max(SIGMA_FLOOR, sigma))
        rows.append({"id": trophy["id"], "q": quantiles(med, sigma)})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            forecast = forecast_case(case)
        except Exception:
            try:
                case = json.loads(path.read_text(encoding="utf-8"))
                forecast = fallback_rows(case)
            except Exception:
                continue
        out_path = OUTPUT_DIR / f"{path.stem}.out"
        out_path.write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
