"""Capture one screenshot per page of the deployed Streamlit app for the report appendix.

Usage (any Python with playwright installed and Chromium downloaded):
    python report/capture_app.py https://<app>.streamlit.app
Writes reports/figures/app_<n>_<page>.png at 1400 x 900 CSS pixels (device scale 2 for print).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "figures"
PAGES = [
    ("1", "data_explorer", ""),
    ("2", "eda", "EDA"),
    ("3", "preprocessing", "Preprocessing"),
    ("4", "model_training", "Model_Training"),
    ("5", "inference", "Inference"),
    ("6", "evaluation", "Evaluation"),
    ("7", "about", "About"),
]


def settle(page, seconds: float = 6.0) -> None:
    """Streamlit renders over a websocket: wait until the 'running' indicator is gone."""
    deadline = time.time() + seconds
    while time.time() < deadline:
        running = page.locator("[data-testid='stStatusWidget']").count()
        if running == 0:
            break
        time.sleep(0.5)
    time.sleep(1.5)


def main(base_url: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        context = browser.new_context(viewport={"width": 1400, "height": 900}, device_scale_factor=2)
        page = context.new_page()
        for number, slug, route in PAGES:
            url = f"{base_url.rstrip('/')}/{route}"
            page.goto(url, wait_until="domcontentloaded", timeout=120_000)
            page.wait_for_timeout(12_000)  # the Streamlit front end renders over a websocket after load
            settle(page, 20.0)
            target = OUT / f"app_{number}_{slug}.png"
            page.screenshot(path=str(target), full_page=False)
            print(f"{target.name}: {page.title()!r}")
        browser.close()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501")
