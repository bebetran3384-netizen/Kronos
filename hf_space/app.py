"""Kronos x Day Trader: one Hugging Face Space.

Pick a US stock. The app pulls free Yahoo 5-minute bars, runs the day-trader
opening-range breakout rules on the latest session (trader-entry), plans the
stop and target (trader-exit), sizes the trade to your max loss (trader-budget),
and asks the Kronos foundation model for many sampled paths of the next bars.
It counts how many of those paths reach the target before the stop.

Nothing here places an order. It is a research tool, not financial advice.
"""

import datetime
import importlib.util
import os
import sys
import threading

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import gradio as gr
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from model import Kronos, KronosTokenizer, KronosPredictor  # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location("trader_" + name, os.path.join(HERE, "trader", name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


entry = _load("entry")
plan_mod = _load("exit")
size_mod = _load("size")

MODELS = {
    "Kronos-mini (4M, fastest)": ("NeoQuasar/Kronos-mini", "NeoQuasar/Kronos-Tokenizer-2k", 2048),
    "Kronos-small (25M)": ("NeoQuasar/Kronos-small", "NeoQuasar/Kronos-Tokenizer-base", 512),
    "Kronos-base (102M, slow on CPU)": ("NeoQuasar/Kronos-base", "NeoQuasar/Kronos-Tokenizer-base", 512),
}
DEFAULT_MODEL = "Kronos-small (25M)"
STEP = 300  # 5-minute bars
LOOKBACK = 400
_predictors = {}
_lock = threading.Lock()


def predictor_for(label):
    with _lock:
        if label not in _predictors:
            repo, tok_repo, ctx = MODELS[label]
            tok = KronosTokenizer.from_pretrained(tok_repo)
            mdl = Kronos.from_pretrained(repo)
            tok.eval()
            mdl.eval()
            _predictors[label] = KronosPredictor(mdl, tok, device="cpu" if not torch.cuda.is_available() else None,
                                                 max_context=ctx)
        return _predictors[label]


# ---------------------------------------------------------------- data

def fetch(sym):
    """60 days of 5-minute bars from Yahoo, through trader-entry's polite reader."""
    text = entry.yahoo_get(entry.chart_path(sym, "60d", "5m"))
    return entry.parse_chart(text, sym)


def rth_frame(chart):
    """Regular-hours 5-minute bars as a DataFrame indexed by New York time (naive)."""
    rows = []
    for b in chart["bars"]:
        if b["t"] % STEP:
            continue
        et = entry.to_eastern(entry.utc_of(b["t"]))
        ok, _ = entry.trading_day(et.date())
        if not ok:
            continue
        o, c = entry.session(et.date())
        if not (o <= et.time() < c):
            continue
        rows.append({"ts": et, "open": b["o"], "high": b["h"], "low": b["l"], "close": b["c"], "volume": float(b["v"])})
    df = pd.DataFrame(rows).drop_duplicates("ts").set_index("ts").sort_index()
    df["amount"] = df["volume"] * df[["open", "high", "low", "close"]].mean(axis=1)
    return df


def next_slots(last_ts, n):
    """The next n regular-session 5-minute bar starts after last_ts (skips nights, weekends, holidays)."""
    out, t = [], last_ts
    while len(out) < n:
        t = t + datetime.timedelta(seconds=STEP)
        d = t.date()
        ok, _ = entry.trading_day(d)
        if not ok:
            t = datetime.datetime.combine(d + datetime.timedelta(days=1), datetime.time(0, 0))
            continue
        o, c = entry.session(d)
        if t.time() < o:
            t = datetime.datetime.combine(d, o)
        if t.time() >= c:
            t = datetime.datetime.combine(d + datetime.timedelta(days=1), datetime.time(0, 0))
            continue
        out.append(t)
    return out


def eval_time(df):
    """When to evaluate the setup: now if the latest session is today, else that session's close."""
    now = entry.utc_now()
    last_day = df.index[-1].date()
    if entry.to_eastern(now).date() == last_day:
        return now, last_day
    _, c = entry.session(last_day)
    return entry.eastern_to_utc(datetime.datetime.combine(last_day, c)), last_day


# ---------------------------------------------------------------- the rules

def run_rules(chart, now_utc, or_minutes, target_r, budget, max_loss):
    res = entry.evaluate(chart, now_utc, setup="orb", or_minutes=int(or_minutes), interval="5m")
    orr = res.get("opening_range")
    out = {"entry": res, "plan": None, "size": None, "hypothetical": False, "error": None}
    if not orr or not res.get("level"):
        out["error"] = "No opening range yet: %s" % (res.get("reason") or res.get("status"))
        return out
    if res["status"] == "triggered":
        px = res["trigger"]["price"]
    else:
        px = round(orr["high"] + 0.01, 2)  # what the entry would be on a breakout
        out["hypothetical"] = True
    try:
        c = plan_mod.compute(px, "or_mid", None, "r", plan_mod.D(target_r), or_high=orr["high"], or_low=orr["low"])
    except plan_mod.PlanError as e:
        out["error"] = "Exit plan refused: %s" % e
        return out
    plan = {"symbol": res["symbol"], "side": "buy", "date": res["date"], "setup": "orb",
            "entry": float(c["entry"]), "stop": float(c["stop"]), "target": float(c["target"]),
            "stop_label": c["stop_label"], "target_label": c["target_label"],
            "risk_per_share": float(c["risk"]), "reward_risk": float(c["reward_risk"]),
            "entry_time": (res.get("trigger") or {}).get("time")}
    out["plan"] = plan
    try:
        out["size"] = size_mod.size_trade(plan, {"max_loss_per_trade_usd": float(max_loss)}, budget=float(budget))
    except size_mod.SizeError as e:
        out["error"] = "Sizing refused: %s" % e
    return out


# ---------------------------------------------------------------- Kronos

def forecast(df_ctx, pred_len, paths, model_label, temperature, top_p, seed):
    pred = predictor_for(model_label)
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    ctx = df_ctx.iloc[-LOOKBACK:]
    x_ts = pd.Series(ctx.index)
    y_ts = pd.Series(next_slots(ctx.index[-1], int(pred_len)))
    cols = ["open", "high", "low", "close", "volume", "amount"]
    n = int(paths)
    with torch.no_grad():
        outs = pred.predict_batch([ctx[cols]] * n, [x_ts] * n, [y_ts] * n, pred_len=int(pred_len),
                                  T=float(temperature), top_p=float(top_p), sample_count=1, verbose=False)
    return y_ts, outs


def path_odds(outs, stop, target):
    """Per sampled path: did the low hit the stop or the high hit the target first?
    A bar that touches both counts as the stop (the cautious read)."""
    t = s = 0
    for p in outs:
        hit = None
        for hi, lo in zip(p["high"].values, p["low"].values):
            if lo <= stop:
                hit = "stop"
                break
            if hi >= target:
                hit = "target"
                break
        t += hit == "target"
        s += hit == "stop"
    n = len(outs)
    return {"target": t, "stop": s, "neither": n - t - s, "n": n}


def actual_outcome(df_after, stop, target):
    for _, r in df_after.iterrows():
        if r["low"] <= stop:
            return "stop", r.name
        if r["high"] >= target:
            return "target", r.name
    return "neither", None


# ---------------------------------------------------------------- the page

def chart_fig(df_show, y_ts, outs, plan, orr, actual=None, title="", day=None):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df_show.index, open=df_show["open"], high=df_show["high"], low=df_show["low"],
                                 close=df_show["close"], name="Price", increasing_line_color="#26a69a",
                                 decreasing_line_color="#ef5350"))
    if actual is not None and len(actual):
        fig.add_trace(go.Candlestick(x=actual.index, open=actual["open"], high=actual["high"], low=actual["low"],
                                     close=actual["close"], name="What happened", opacity=0.35,
                                     increasing_line_color="#26a69a", decreasing_line_color="#ef5350"))
    if outs:
        closes = np.stack([p["close"].values for p in outs])
        for row in closes[: min(len(closes), 30)]:
            fig.add_trace(go.Scatter(x=y_ts, y=row, mode="lines", line=dict(width=1, color="rgba(124,92,255,0.18)"),
                                     hoverinfo="skip", showlegend=False))
        q = np.percentile(closes, [10, 50, 90], axis=0)
        fig.add_trace(go.Scatter(x=y_ts, y=q[2], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=y_ts, y=q[0], mode="lines", line=dict(width=0), fill="tonexty",
                                 fillcolor="rgba(124,92,255,0.15)", name="Kronos 10–90%"))
        fig.add_trace(go.Scatter(x=y_ts, y=q[1], mode="lines", line=dict(width=2, color="#7c5cff"),
                                 name="Kronos median"))
    if orr and day is not None:
        o, c = entry.session(day)
        o_dt = datetime.datetime.combine(day, o)
        fig.add_shape(type="rect", x0=o_dt, x1=o_dt + datetime.timedelta(minutes=orr["minutes"]), y0=orr["low"],
                      y1=orr["high"], fillcolor="rgba(255,193,7,0.35)", line_width=0, layer="below")
    if plan:
        for y, name, color in ((plan["entry"], "entry", "#2196f3"), (plan["stop"], "stop", "#ef5350"),
                               (plan["target"], "target", "#26a69a")):
            fig.add_hline(y=y, line_dash="dash", line_color=color, annotation_text="%s $%.2f" % (name, y),
                          annotation_position="top left")
    fig.update_layout(title=title, height=560, margin=dict(l=10, r=10, t=50, b=10), xaxis_rangeslider_visible=False,
                      legend=dict(orientation="h", y=-0.08, x=0), template="plotly_white")
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"]), dict(bounds=[16, 9.5], pattern="hour")])
    return fig


def money(x):
    return "$%s" % format(x, ",.2f")


def report(sym, rules, odds, anchor_note, actual=None, last_close=None):
    e, p, s = rules["entry"], rules["plan"], rules["size"]
    lines = ["## %s — session %s" % (sym, e["date"])]
    orr = e.get("opening_range")
    if orr:
        lines.append("**Opening range (%d min):** high %s, low %s" % (orr["minutes"], money(orr["high"]), money(orr["low"])))
    status = {"triggered": "✅ Breakout triggered", "waiting": "⏳ Waiting for a close above the range high",
              "expired": "⌛ No breakout before the entry cutoff"}.get(e["status"], e["status"])
    lines.append("**Entry (trader-entry):** %s — %s" % (status, e.get("reason", "")))
    if rules.get("error") and not p:
        lines.append("\n> %s" % rules["error"])
        return "\n\n".join(lines)
    if p:
        hyp = " *(if it breaks out — not triggered)*" if rules["hypothetical"] else ""
        lines.append("**Plan (trader-exit)%s:** entry %s · stop %s (%s) · target %s (%s) · risk %s/share · %s:1"
                     % (hyp, money(p["entry"]), money(p["stop"]), p["stop_label"], money(p["target"]),
                        p["target_label"], money(p["risk_per_share"]), p["reward_risk"]))
    if s:
        if s["refusal"]:
            lines.append("**Size (trader-budget):** ❌ %s" % s["refusal"])
        else:
            lines.append("**Size (trader-budget):** %d shares = %s, max loss at the stop %s (bound by %s)"
                         % (s["qty"], money(s["cost"]), money(s["max_loss"]), s["bound_by"]))
    elif rules.get("error"):
        lines.append("> %s" % rules["error"])
    if odds and last_close is not None and p and not (p["stop"] < last_close < p["target"]):
        lines.append("> ⚠️ The last close %s is already %s, so every path starts past that level: the split below says "
                     "little." % (money(last_close), "at or under the stop" if last_close <= p["stop"] else "at or over the target"))
    if odds:
        n = odds["n"]
        lines.append("### Kronos sampled paths (%d) — %s" % (n, anchor_note))
        lines.append("| reach target first | hit stop first | neither in the window |\n|---|---|---|\n| %d (%.0f%%) | %d (%.0f%%) | %d (%.0f%%) |"
                     % (odds["target"], 100.0 * odds["target"] / n, odds["stop"], 100.0 * odds["stop"] / n,
                        odds["neither"], 100.0 * odds["neither"] / n))
        if n < 20:
            lines.append("_%d paths is a small sample: treat the split as rough._" % n)
    if actual:
        lines.append("**What actually happened after the entry bar:** %s" % actual)
    lines.append("\n---\n_Research only. Not financial advice. No order is placed. A stop is a plan, not a guarantee. "
                 "Yahoo data is free and delayed. Kronos is a generative model: its paths are samples, not predictions you "
                 "can bank on._")
    return "\n\n".join(lines)


def analyze(sym, model_label, anchor, pred_len, paths, or_minutes, target_r, budget, max_loss,
            temperature, top_p, seed, progress=gr.Progress()):
    sym = (sym or "").strip().upper()
    if not sym:
        raise gr.Error("Type a ticker, e.g. TSLA")
    progress(0.05, desc="Fetching Yahoo 5-minute bars")
    try:
        chart = fetch(sym)
    except entry.EntryError as e:
        raise gr.Error(str(e))
    df = rth_frame(chart)
    if len(df) < 120:
        raise gr.Error("Only %d regular-hours bars for %s: need at least 120" % (len(df), sym))
    now_utc, day = eval_time(df)
    rules = run_rules(chart, now_utc, or_minutes, target_r, budget, max_loss)
    e, p = rules["entry"], rules["plan"]

    replay = anchor.startswith("Entry bar") and e["status"] == "triggered"
    if replay:
        _, close_t = entry.session(day)
        trig_end = datetime.datetime.combine(day, entry.parse_hhmm(e["trigger"]["close_time"], "t"))
        ctx = df[df.index < trig_end]
        after = df[(df.index >= trig_end) & (df.index < datetime.datetime.combine(day, close_t))]
        n_pred = max(1, min(int(pred_len), len(after))) if len(after) else int(pred_len)
        note = "replay from the entry bar (%s), %d bars" % (e["trigger"]["time"], n_pred)
    else:
        ctx, after, n_pred = df, None, int(pred_len)
        note = "from the latest bar (%s), next %d bars" % (df.index[-1].strftime("%b %d %H:%M"), n_pred)
        if anchor.startswith("Entry bar"):
            note += " — no breakout this session, so no entry bar to replay from"

    progress(0.25, desc="Loading %s and sampling %d paths" % (model_label.split(" ")[0], int(paths)))
    y_ts, outs = forecast(ctx, n_pred, paths, model_label, temperature, top_p, seed)
    odds = path_odds(outs, p["stop"], p["target"]) if p else None

    actual_txt, actual_df = None, None
    if replay and after is not None and len(after):
        actual_df = after.iloc[:n_pred]
        what, when = actual_outcome(after, p["stop"], p["target"])
        actual_txt = {"target": "reached the target at %s" % (when.strftime("%H:%M") if when is not None else ""),
                      "stop": "hit the stop at %s" % (when.strftime("%H:%M") if when is not None else ""),
                      "neither": "neither by the close (last close %s)" % money(after["close"].iloc[-1])}[what]

    show = ctx.iloc[-156:]  # about two sessions
    fig = chart_fig(show, y_ts, outs, p, e.get("opening_range"), actual_df,
                    "%s 5-minute bars + Kronos (%s)" % (sym, model_label.split(" ")[0]), day)
    progress(1.0)
    return fig, report(sym, rules, odds, note, actual_txt, float(ctx["close"].iloc[-1]))


with gr.Blocks(title="Kronos Day Trader") as demo:
    gr.Markdown("# Kronos × Day Trader\nOpening-range breakout rules from **day-trader-skill**, path forecasts from the "
                "**Kronos** foundation model. Free Yahoo data. Paper research only: no orders, not financial advice.")
    with gr.Row():
        with gr.Column(scale=1, min_width=280):
            sym = gr.Textbox(label="Ticker", value="TSLA")
            model_label = gr.Dropdown(list(MODELS), value=DEFAULT_MODEL, label="Kronos model")
            anchor = gr.Radio(["Latest bar (look ahead)", "Entry bar (replay the breakout)"],
                              value="Entry bar (replay the breakout)", label="Forecast from")
            pred_len = gr.Slider(6, 78, value=36, step=1, label="Bars to forecast (5 min each)")
            paths = gr.Slider(4, 40, value=16, step=1, label="Sampled paths")
            with gr.Accordion("Trade rules", open=False):
                or_minutes = gr.Dropdown([5, 15, 30], value=15, label="Opening range (minutes)")
                target_r = gr.Slider(1, 4, value=2, step=0.5, label="Target (R multiple)")
                budget = gr.Number(value=5000, label="Budget ($)")
                max_loss = gr.Number(value=50, label="Max loss per trade ($)")
            with gr.Accordion("Sampling", open=False):
                temperature = gr.Slider(0.1, 2.0, value=1.0, step=0.1, label="Temperature")
                top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="top_p")
                seed = gr.Number(value=19, precision=0, label="Seed")
            go_btn = gr.Button("Analyze", variant="primary")
        with gr.Column(scale=3):
            plot = gr.Plot()
            out = gr.Markdown()
    go_btn.click(analyze, [sym, model_label, anchor, pred_len, paths, or_minutes, target_r, budget, max_loss,
                           temperature, top_p, seed], [plot, out], concurrency_limit=1)

if __name__ == "__main__":
    demo.queue().launch(theme=gr.themes.Soft())
