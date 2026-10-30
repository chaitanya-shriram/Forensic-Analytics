"""End-to-end UI test: drives the built console in Chromium against a running backend.

Usage: python e2e/test_ui.py [base_url]   (backend must be running)
Screenshots are written to e2e/screenshots/.
"""
import sys
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
OUT = Path(__file__).parent / "screenshots"
OUT.mkdir(exist_ok=True)


def main():
    errors = []
    with sync_playwright() as p:
        b = p.chromium.launch(args=["--use-gl=angle", "--enable-webgl", "--ignore-gpu-blocklist"])
        page = b.new_page(viewport={"width": 1560, "height": 1000})
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: m.type == "error" and errors.append(m.text))

        page.goto(BASE)
        expect(page.get_by_text("What needs review")).to_be_visible(timeout=900_000)
        expect(page.locator("table.queue tbody tr").first).to_be_visible(timeout=30_000)
        expect(page.locator(".graph3d canvas, .main canvas").first).to_be_visible(timeout=30_000)
        page.wait_for_timeout(3000)
        page.screenshot(path=OUT / "1_triage.png", full_page=True)

        page.get_by_role("link", name="How scoring works", exact=True).click()
        expect(page.get_by_text("Cut out the neighbourhood")).to_be_visible(timeout=60_000)
        assert page.locator(".subgraph-svg circle").count() > 3, "subgraph not drawn"
        page.screenshot(path=OUT / "1b_how_scoring_works.png", full_page=True)

        page.get_by_role("link", name="Dataset", exact=True).click()
        expect(page.get_by_text("Cleaning pipeline")).to_be_visible()
        page.wait_for_timeout(800)
        page.screenshot(path=OUT / "2_dataset.png", full_page=True)

        page.get_by_role("link", name="Models", exact=True).click()
        expect(page.get_by_text("Test-set comparison")).to_be_visible()
        page.wait_for_timeout(800)
        page.screenshot(path=OUT / "3_models.png", full_page=True)

        page.get_by_role("link", name="Fraud rings", exact=True).click()
        expect(page.get_by_text("Ring network · 3D")).to_be_visible(timeout=20_000)
        expect(page.locator(".graph3d canvas")).to_be_visible(timeout=30_000)
        page.wait_for_timeout(3500)
        page.screenshot(path=OUT / "4_rings.png", full_page=True)

        page.get_by_role("link", name="Accounts", exact=True).click()
        expect(page.locator("table.table tbody tr").first).to_be_visible()
        page.locator("table.table tbody tr a.link").first.click()
        expect(page.get_by_text("Why this score?")).to_be_visible(timeout=30_000)
        expect(page.get_by_text("Outer GNN · cross-bank attention")).to_be_visible()
        page.wait_for_timeout(3500)
        page.screenshot(path=OUT / "5_account.png", full_page=True)

        page.get_by_role("link", name="Live monitor", exact=True).click()
        expect(page.get_by_role("heading", name="Live monitor")).to_be_visible()
        page.get_by_role("button", name="stream").click()
        expect(page.get_by_role("button", name="Stop")).to_be_visible()
        n0 = page.locator("table.table tbody tr").count()
        page.wait_for_timeout(15_000)
        n1 = page.locator("table.table tbody tr").count()
        assert n1 > n0, f"transaction feed did not grow ({n0} -> {n1})"
        page.screenshot(path=OUT / "6_live.png", full_page=True)
        page.get_by_role("button", name="Stop").click()

        page.get_by_role("link", name="Privacy & federation", exact=True).click()
        expect(page.get_by_text("Never leaves a bank")).to_be_visible()
        page.wait_for_timeout(600)
        page.screenshot(path=OUT / "7_privacy.png", full_page=True)
        b.close()
    real = [e for e in errors if "favicon" not in e and "WebGL" not in e and "GPU stall" not in e]
    assert not real, f"browser errors: {real}"
    print("E2E OK; screenshots in", OUT)


if __name__ == "__main__":
    main()
