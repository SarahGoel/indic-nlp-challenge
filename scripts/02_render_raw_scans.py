from pathlib import Path
import pymupdf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = PROJECT_ROOT / "data" / "raw_scans" / "BRIGU_V5_SOURCE.pdf"
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw_scans"

DPI = 300
ZOOM = DPI / 72

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(PDF_PATH)

    print(f"Source: {PDF_PATH}")
    print(f"Pages: {doc.page_count}")
    print(f"Rendering: {DPI} DPI")

    for index, page in enumerate(doc):
        page_id = f"BRIGU_V5_P{index + 1:04d}"
        output_path = OUTPUT_DIR / f"{page_id}.png"

        if output_path.exists():
            print(f"[SKIP] {page_id} already exists")
            continue

        matrix = pymupdf.Matrix(ZOOM, ZOOM)
        pix = page.get_pixmap(
            matrix=matrix,
            colorspace=pymupdf.csRGB,
            alpha=False,
        )

        pix.save(output_path)

        print(
            f"[{index + 1:03d}/{doc.page_count}] "
            f"{page_id}.png"
        )

    doc.close()
    print("Rendering complete.")

if __name__ == "__main__":
    main()
