"""Recheck national news after removing municipality pseudoreplication."""
from pathlib import Path
from news_ablation_global import forecast, load_panel

if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "outputs/research"
    out.mkdir(parents=True, exist_ok=True)
    print(forecast(load_panel(),outdir=out).to_string(index=False))
