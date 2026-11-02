---
title: Kronos Day Trader
emoji: 📈
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 6.31.0
app_file: app.py
pinned: false
license: mit
short_description: ORB day-trade rules + Kronos sampled price paths
---

# Kronos × Day Trader

One web app built from two repos:

- **[Kronos](https://github.com/bebetran3384-netizen/Kronos)**, the foundation model for financial
  candlesticks (`model/`, weights from `NeoQuasar/*` on the Hugging Face Hub).
- **[day-trader-skill](https://github.com/bebetran3384-netizen/day-trader-skill)**: the opening-range
  breakout entry rule (`trader/entry.py`), the stop and target planner (`trader/exit.py`) and the
  position sizer (`trader/size.py`), used unchanged.

## What it does

1. Pulls 60 days of free Yahoo 5-minute bars for the ticker.
2. **Entry:** runs trader-entry on the latest session: the 15-minute opening range, then the first
   bar that closes above the range high.
3. **Plan:** trader-exit puts the stop at the range midpoint and the target at 2R. With no
   breakout, it plans the hypothetical entry one cent over the range high.
4. **Size:** trader-budget sizes to your budget and max loss per trade.
5. **Kronos:** samples N independent paths of the next bars, draws the median and 10–90% band,
   and counts the paths that hit the target before the stop.
   - *Entry bar (replay):* forecasts from the breakout bar with only the bars before it, then shows
     what actually happened next and whether the trade hit its stop or target.
   - *Latest bar:* forecasts forward from the last bar.

It never places an order. It's research only, not financial advice.

## Deploy to Hugging Face Spaces

Create a new **Gradio** Space, then push the contents of this folder to it:

```bash
git clone https://huggingface.co/spaces/<you>/kronos-day-trader
cp -r hf_space/. kronos-day-trader/
cd kronos-day-trader && git add -A && git commit -m "Kronos Day Trader" && git push
```

The free CPU hardware is enough. Kronos-mini takes about 10 seconds, Kronos-small 30 seconds for
16 paths of 36 bars. Kronos-base needs a GPU Space to run at a comfortable speed.

## Run locally

```bash
pip install -r requirements.txt gradio
python app.py   # http://127.0.0.1:7860
```

## Keeping the vendored code fresh

`model/` and `trader/` are copies, so the Space folder is self-contained. To refresh them:

```bash
./sync.sh /path/to/day-trader-skill
```

Licenses: Kronos MIT (`model/LICENSE`), day-trader-skill MIT (`trader/LICENSE`).
