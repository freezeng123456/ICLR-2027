from pathlib import Path
import argparse
import hashlib
import json
import subprocess

from PIL import Image, ImageDraw
from pypdf import PdfReader


root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--pdf", type=Path, default=root / "work/paper-build/main.pdf")
parser.add_argument("--output", type=Path, default=root / "work/paper-qa/final")
args = parser.parse_args()
pdf = args.pdf.resolve()
output = args.output.resolve()
output.mkdir(parents=True, exist_ok=True)
renderer = "/Users/zenghang/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/override/pdftoppm"
subprocess.run([renderer, "-scale-to", "1400", "-png", str(pdf), str(output / "page")], check=True)
reader = PdfReader(pdf)
for start in range(0, len(reader.pages), 12):
    sheet = Image.new("RGB", (1200, 1320), "#dddddd")
    draw = ImageDraw.Draw(sheet)
    for offset in range(12):
        index = start + offset
        if index >= len(reader.pages):
            break
        path = output / f"page-{index + 1:02d}.png"
        with Image.open(path) as image:
            image.thumbnail((292, 410))
            left, top = (offset % 4) * 300 + 4, (offset // 4) * 440 + 22
            sheet.paste(image, (left, top))
        draw.text(((offset % 4) * 300 + 8, (offset // 4) * 440 + 5), f"Page {index + 1}", fill="black")
    sheet.save(output / f"sheet-{start // 12 + 1}.png")
texts = [page.extract_text() for page in reader.pages]
if any("??" in text for text in texts):
    raise RuntimeError("unresolved PDF reference marker")
fonts = set()
for page in reader.pages:
    for font in page["/Resources"]["/Font"].get_object().values():
        fonts.add(str(font.get_object()["/BaseFont"]))
report = {
    "pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
    "pages": len(texts),
    "page_resource_fonts": sorted(fonts),
    "rendered_all_pages": True,
    "unresolved_reference_markers": False,
    "page_text_characters": [len(text) for text in texts],
    "references_start_pages": [index + 1 for index, text in enumerate(texts) if "REFERENCES" in text],
    "visual_inspection": "pending; rendering alone is not visual acceptance",
}
(output / "render-report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2), flush=True)
