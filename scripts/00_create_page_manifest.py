from pathlib import Path
import csv

TOTAL_PAGES = 888
OUTPUT = Path("data/raw_scans/page_manifest.csv")


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "page_id",
                "pdf_page_number",
                "printed_page_number",
                "page_type",
                "content_status",
                "khand",
                "notes",
            ]
        )

        for pdf_page_number in range(1, TOTAL_PAGES + 1):
            page_id = f"BRIGU_V5_P{pdf_page_number:04d}"

            page_type = "unknown"
            content_status = "pending"
            khand = ""
            notes = ""

            if pdf_page_number == 1:
                page_type = "cover"
                content_status = "excluded"
                notes = "Red front cover containing title."

            elif pdf_page_number == 2:
                page_type = "blank"
                content_status = "excluded"
                notes = "Blank page immediately after front cover."

            elif pdf_page_number >= 885:
                page_type = "non_content"
                content_status = "excluded"
                notes = "Final four pages contain no useful text; exact page characteristics documented in SOURCE_FILE_RECORD.md."

            writer.writerow(
                [
                    page_id,
                    pdf_page_number,
                    "",
                    page_type,
                    content_status,
                    khand,
                    notes,
                ]
            )


if __name__ == "__main__":
    main()