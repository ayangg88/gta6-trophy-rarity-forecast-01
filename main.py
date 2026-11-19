"""Forecast GTA VI PlayStation 5 trophy earn rates on Exophase.

For each case the program reads every trophy's history of reads and forecasts
five quantiles of its earn rate at the case's target, as one JSON object per
case. Standard library only, deterministic, no network.

Model: a trophy's earn rate is earners over tracked players, r = E / N.
Earners and the tracked population are forecast separately, because the
population grows faster than any trophy's earners early on and dilutes the
rates: both log(E) and log(N) are fit against x = ln(1 + hours since the
game's release) by recency-weighted least squares, then extrapolated to the
target. Each trophy's earner slope is shrunk toward its tier's median slope,
so noisy rare trophies borrow strength from their tier. The spread around the
median combines a random-walk term, the fit's residual and the slope's
uncertainty. A flat history (no movement at all, as in the practice cases)
falls back to the last rate with a dilution shift and a wide spread, because
a refresh can move every rate a long way from a flat page.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]


def parse_time(value):
    """Parse an ISO-8601 timestamp (or a plain date) to an aware datetime."""
    text = str(value).strip()
    if len(text) == 10:
        text += "T00:00:00+00:00"
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    moment = datetime.fromisoformat(text)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment


def norm_ppf(p):
    """Inverse standard normal CDF (Acklam's approximation, ~1e-9 accurate)."""
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285984469534e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1 - p))


def inv_logit(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    """Round to 4 decimals, half up; whole numbers come back as ints."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def wls(xs, ys, ws):
    """Weighted least squares of y on x. Returns (intercept, slope, resid_std, slope_se)."""
    sw = sum(ws)
    if sw <= 0 or len(xs) < 2:
        return None
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return (my, 0.0, 0.0, 0.0)
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    n = len(xs)
    resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    dof = max(n - 2, 1)
    resid_var = sum(w * r * r for w, r in zip(ws, resid)) / (sw * dof / n) if n > 2 else 0.0
    resid_std = math.sqrt(max(resid_var, 0.0))
    slope_se = math.sqrt(resid_var / sxx) if sxx > 0 else 0.0
    return (intercept, slope, resid_std, slope_se)


def median(values):
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    if n % 2:
        return ordered[n // 2]
    return 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])


def forecast_case(case):
    game = case.get("game", {})
    release = parse_time(game.get("release_date", "2026-11-19"))
    target = parse_time(case["target"])
    horizon_days = case["horizon_minutes"] / 1440.0
    x_target = math.log1p((target - release).total_seconds() / 3600.0)
    trophies = case["trophies"]
    history = case["history"]
    levels = case.get("levels", LEVELS)
    zs = [norm_ppf(p) for p in levels]

    # Build per-read time points and the population series.
    reads = []
    for entry in history:
        at = parse_time(entry["at"])
        hours = (at - release).total_seconds() / 3600.0
        if hours < 0:
            hours = 0.0
        reads.append({
            "x": math.log1p(hours),
            "at": at,
            "n": float(entry.get("players_tracked") or 0),
            "rates": entry.get("rates") or [],
        })
    last = reads[-1]
    x_last = last["x"]
    # Recency weights: 24-hour half-life from the last read.
    for read in reads:
        age_h = (last["at"] - read["at"]).total_seconds() / 3600.0
        read["w"] = 0.5 ** (age_h / 24.0)

    # Population fit: log(N) on x.
    n_fit = None
    xs_n = [r["x"] for r in reads if r["n"] > 0]
    ys_n = [math.log(r["n"]) for r in reads if r["n"] > 0]
    ws_n = [r["w"] for r in reads if r["n"] > 0]
    if len(xs_n) >= 2:
        n_fit = wls(xs_n, ys_n, ws_n)
    if n_fit is None:
        n_target = last["n"] if last["n"] > 0 else 1.0
        n_slope = 0.0
    else:
        n_target = math.exp(n_fit[0] + n_fit[1] * x_target)
        n_slope = n_fit[1]
    n_target = max(n_target, 1.0)

    # Per-trophy earner fits, then tier shrinkage of the slopes.
    tier_slopes = {}
    fits = []
    for i, trophy in enumerate(trophies):
        xs, ys, ws = [], [], []
        for read in reads:
            rates = read["rates"]
            if i >= len(rates) or rates[i] is None:
                continue
            rate = float(rates[i])
            earners = max(rate / 100.0 * read["n"], 0.5) if read["n"] > 0 else 0.5
            xs.append(read["x"])
            ys.append(math.log(earners))
            ws.append(read["w"])
        fit = wls(xs, ys, ws) if len(xs) >= 2 else None
        fits.append(fit)
        if fit is not None:
            tier_slopes.setdefault(trophy.get("tier", "bronze"), []).append(fit[1])
    tier_median = {t: median(v) for t, v in tier_slopes.items()}

    rows = []
    for i, trophy in enumerate(trophies):
        last_rate = last["rates"][i] if i < len(last["rates"]) and last["rates"][i] is not None else None
        if last_rate is None:
            last_rate = 0.0
        # The trophy everyone already has stays at 100%.
        if last_rate >= 99.99:
            rows.append({"id": trophy["id"], "q": [99.9, 100, 100, 100, 100]})
            continue
        fit = fits[i]
        flat = fit is None or (abs(fit[1]) < 1e-9 and abs(n_slope) < 1e-9)
        if flat:
            # No movement to learn from: the page has not refreshed yet, and
            # the next refresh brings a wave of new tracked players that
            # dilutes every rate (launch night: 638 -> 3,868 tracked, rates
            # fell to roughly a fifth). Expect a large dilution while the
            # tracked population is still tiny, and a wide spread.
            if last["n"] > 0 and last["n"] < 10000:
                dilution = 0.3
            else:
                dilution = 1.0 / math.sqrt(1.0 + horizon_days)
            med = clamp_pct(last_rate * dilution) if last_rate > 0 else 0.05
            sigma = min(2.5, max(0.5, 1.2 * math.sqrt(horizon_days)))
        else:
            slope = fit[1]
            tier_med = tier_median.get(trophy.get("tier", "bronze"), slope)
            n_pts = len([1 for read in reads
                         if i < len(read["rates"]) and read["rates"][i] is not None])
            shrink_w = n_pts / (n_pts + 4.0)
            slope = shrink_w * slope + (1 - shrink_w) * tier_med
            e_target = math.exp(fit[0] + slope * x_target)
            med = clamp_pct(100.0 * e_target / n_target)
            resid = fit[2]
            slope_unc = fit[3] * max(x_target - x_last, 0.0)
            sigma = math.sqrt((0.5 * math.sqrt(horizon_days)) ** 2
                              + resid ** 2 + slope_unc ** 2)
            sigma = min(2.5, max(0.35, sigma))
        lg = logit(med)
        qs = [round4(clamp_pct(inv_logit(lg + z * sigma))) for z in zs]
        qs = sorted(qs)
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def fallback_case(case):
    """Last read's rates with a wide spread; used only if forecasting fails."""
    last = case["history"][-1]
    horizon_days = case["horizon_minutes"] / 1440.0
    sigma = min(2.5, max(0.35, 1.0 * math.sqrt(horizon_days)))
    zs = [norm_ppf(p) for p in case.get("levels", LEVELS)]
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rate = last["rates"][i] if i < len(last["rates"]) and last["rates"][i] is not None else 0.0
        if rate >= 99.99:
            rows.append({"id": trophy["id"], "q": [99.9, 100, 100, 100, 100]})
            continue
        lg = logit(rate if rate > 0 else 0.05)
        qs = sorted(round4(clamp_pct(inv_logit(lg + z * sigma))) for z in zs)
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case)
        except Exception:
            forecast = fallback_case(case)
        out = json.dumps(forecast, separators=(",", ":")) + "\n"
        if len(out.encode("utf-8")) > 65536:
            forecast = fallback_case(case)
            out = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(out, encoding="utf-8")


if __name__ == "__main__":
    main()
