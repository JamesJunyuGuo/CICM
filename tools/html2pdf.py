#!/usr/bin/env python
"""Render an HTML figure to a *vector* PDF with headless Firefox (WebDriver BiDi print).

Usage: html2pdf.py FIG.html OUT.pdf WIDTH_PX HEIGHT_PX [--png OUT.png]

The page is printed at exactly WIDTH_PX x HEIGHT_PX CSS pixels (96 px/in), zero margins,
backgrounds on, no shrink-to-fit. Text stays selectable and scales without rasterization.
"""
import argparse
import base64
import os
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "pylib"))
from selenium import webdriver  # noqa: E402
from selenium.webdriver.common.print_page_options import PrintOptions  # noqa: E402
from selenium.webdriver.firefox.options import Options  # noqa: E402
from selenium.webdriver.firefox.service import Service  # noqa: E402

CM_PER_PX = 2.54 / 96.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("html")
    ap.add_argument("pdf")
    ap.add_argument("width", type=float)
    ap.add_argument("height", type=float)
    ap.add_argument("--png", default=None, help="also rasterize a preview PNG (2x) via pdftoppm")
    a = ap.parse_args()

    opts = Options()
    opts.add_argument("-headless")
    # allow file:// fonts declared via @font-face and keep print colours exact
    opts.set_preference("print.print_bgcolor", True)
    opts.set_preference("print.print_bgimages", True)
    opts.set_preference("print.always_print_silent", True)
    opts.set_preference("layout.css.prefers-color-scheme.content-override", 1)
    gd = pathlib.Path(__file__).resolve().parent / "bin" / "geckodriver"
    drv = webdriver.Firefox(service=Service(str(gd)), options=opts)
    try:
        drv.set_window_size(int(a.width) + 20, int(a.height) + 120)
        drv.get(pathlib.Path(a.html).resolve().as_uri())
        drv.execute_script("return document.fonts.ready.then(()=>true)")
        # report the laid-out height of the page content so the canvas can be sized exactly
        wrap_h = drv.execute_script(
            "const w=document.querySelector('.wrap')||document.body;"
            "return Math.ceil(Math.max(w.scrollHeight, w.getBoundingClientRect().bottom));")
        print(f"content height: {wrap_h} px (canvas {a.height:g} px)"
              + ("  <-- OVERFLOW" if wrap_h > a.height else ""))
        po = PrintOptions()
        po.page_width = a.width * CM_PER_PX
        po.page_height = a.height * CM_PER_PX
        po.margin_top = po.margin_bottom = po.margin_left = po.margin_right = 0.0
        po.background = True
        po.shrink_to_fit = False
        po.scale = 1.0
        data = drv.print_page(po)
    finally:
        drv.quit()
    out = pathlib.Path(a.pdf)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(base64.b64decode(data))
    if a.png:
        stem = str(pathlib.Path(a.png).with_suffix(""))
        subprocess.run(["pdftoppm", "-r", "192", "-png", "-singlefile", str(out), stem], check=True)
    print(f"wrote {out} ({out.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
