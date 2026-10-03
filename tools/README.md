# tools

`html2pdf.py` renders an HTML figure to a vector PDF with headless Firefox (WebDriver BiDi print).

Setup (once):

```bash
pip install --target tools/pylib "selenium==4.25.0"
curl -sSL https://github.com/mozilla/geckodriver/releases/download/v0.36.0/geckodriver-v0.36.0-linux64.tar.gz | tar xz -C tools/bin
```

Usage:

```bash
MOZ_HEADLESS=1 python tools/html2pdf.py fig.html out.pdf 1000 460 --png preview.png
```
