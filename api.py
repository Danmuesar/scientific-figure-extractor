import io
import json
import tempfile
import zipfile
from pathlib import Path
from typing import List

import fitz
from starlette.background import BackgroundTask
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse


params = {
    "min_shape_area": 10,
    "max_page_area_frac": 0.1,
    "caption_gap": 3.175,
    "iterations": 9,
    "expansion_step": 1.175,
    "max_expansion": 2.0,
    "initial_w_margin_factor": 0.50,
    "initial_h_margin_factor": 0.175,
    "band_min_area": 10.0,
    "band_aspect_low": 0.2,
    "band_aspect_high": 5.0,
    "label_margin": 3,
    "label_max_words": 8,
    "label_max_fontsize": 20.0,
}


app = FastAPI(title="Figure Extractor API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_FILE = BASE_DIR / "figure_extractor_local.html"
REPORT_IMAGE_PREVIEW_LIMIT = 50
UPLOAD_CHUNK_SIZE = 1024 * 1024


def is_caption(text: str) -> bool:
    normalized = text.strip().lower()
    return normalized.startswith(("figure", "fig.", "fig ", "plate"))


def merge_rectangles(rects: List[fitz.Rect]) -> fitz.Rect:
    return fitz.Rect(
        min(r.x0 for r in rects),
        min(r.y0 for r in rects),
        max(r.x1 for r in rects),
        max(r.y1 for r in rects),
    )


def is_ignorable_shape(shape, page_area: float) -> bool:
    rect = fitz.Rect(shape.get("rect", (0, 0, 0, 0)))
    area = rect.width * rect.height
    if area <= 0:
        return True
    if area < params["min_shape_area"]:
        return True
    if area > page_area * params["max_page_area_frac"]:
        return True
    width = shape.get("width")
    if isinstance(width, (int, float)) and width == 0:
        return True
    return False


def clamp_rect_to_page(rect: fitz.Rect, page_rect: fitz.Rect) -> fitz.Rect:
    return fitz.Rect(
        max(rect.x0, page_rect.x0),
        max(rect.y0, page_rect.y0),
        min(rect.x1, page_rect.x1),
        min(rect.y1, page_rect.y1),
    )


def expand_with_labels_and_shapes(
    figure_rect: fitz.Rect,
    text_blocks: List[dict],
    vector_rects: List[fitz.Rect],
    captions: List[fitz.Rect],
) -> fitz.Rect:
    expanded = fitz.Rect(figure_rect)
    w_margin = figure_rect.width * params["initial_w_margin_factor"]
    h_margin = figure_rect.height * params["initial_h_margin_factor"]

    for _ in range(params["iterations"]):
        new_added = False

        top_strip = fitz.Rect(expanded.x0, expanded.y0 - h_margin, expanded.x1, expanded.y0)
        bottom_strip = fitz.Rect(expanded.x0, expanded.y1, expanded.x1, expanded.y1 + h_margin)
        left_strip = fitz.Rect(expanded.x0 - w_margin, expanded.y0, expanded.x0, expanded.y1)
        right_strip = fitz.Rect(expanded.x1, expanded.y0, expanded.x1 + w_margin, expanded.y1)

        for tb in text_blocks:
            tb_rect = tb["rect"]
            if not tb["text"]:
                continue
            if any(tb_rect.intersects(cap) for cap in captions):
                continue
            if tb["words"] > params["label_max_words"]:
                continue
            if tb["max_size"] > params["label_max_fontsize"]:
                continue

            if (
                top_strip.intersects(tb_rect)
                or bottom_strip.intersects(tb_rect)
                or left_strip.intersects(tb_rect)
                or right_strip.intersects(tb_rect)
            ):
                expanded = expanded | tb_rect
                new_added = True

        for vrect in vector_rects:
            if any(vrect.intersects(cap) for cap in captions):
                continue
            if expanded.intersects(vrect):
                continue
            if (
                top_strip.intersects(vrect)
                or bottom_strip.intersects(vrect)
                or left_strip.intersects(vrect)
                or right_strip.intersects(vrect)
            ):
                expanded = expanded | vrect
                new_added = True

        if not new_added:
            break

        w_margin *= params["expansion_step"]
        h_margin *= params["expansion_step"]
        if w_margin > figure_rect.width * params["max_expansion"]:
            break

    expanded = clamp_rect_to_page(expanded, fitz.Rect(0, 0, 1e9, 1e9))
    margin = params.get("label_margin", 0)
    return fitz.Rect(
        expanded.x0 - margin,
        expanded.y0 - margin,
        expanded.x1 + margin,
        expanded.y1 + margin,
    )


def extract_figures_from_document(doc: fitz.Document) -> List[dict]:
    figures = []
    for page_index, page in enumerate(doc):
        page_area = page.rect.width * page.rect.height

        raw_shapes = page.get_drawings()
        vector_shapes = [fitz.Rect(s["rect"]) for s in raw_shapes if not is_ignorable_shape(s, page_area)]

        blocks_dict = page.get_text("dict")["blocks"]
        raster_rects = []
        for block in blocks_dict:
            if block.get("type") == 1:
                raster_rects.append(fitz.Rect(block["bbox"]))

        try:
            for img in page.get_images(full=True):
                xref = img[0]
                try:
                    for rect in page.get_image_rects(xref):
                        image_rect = fitz.Rect(rect)
                        if not any(image_rect == existing for existing in raster_rects):
                            raster_rects.append(image_rect)
                except Exception:
                    pass
        except Exception:
            pass

        shape_items = [{"rect": r, "type": "vector"} for r in vector_shapes]
        shape_items.extend({"rect": r, "type": "raster"} for r in raster_rects)

        text_blocks = []
        captions = []
        caption_text_by_y0 = {}
        for block in blocks_dict:
            if block.get("type") != 0:
                continue

            rect = fitz.Rect(block["bbox"])
            text = "".join(span["text"] for line in block["lines"] for span in line["spans"]).strip()
            if not text:
                continue

            max_size = max(
                (span.get("size", 0) for line in block["lines"] for span in line["spans"]),
                default=0,
            )
            text_blocks.append(
                {
                    "rect": rect,
                    "text": text,
                    "max_size": max_size,
                    "words": len(text.split()),
                }
            )

            if is_caption(text):
                captions.append(rect)
                caption_text_by_y0[rect.y0] = text

        captions.sort(key=lambda rect: rect.y0)
        if not captions:
            continue

        all_vector_rects = [item["rect"] for item in shape_items if item["type"] == "vector"]
        page_figure_index = 0

        for idx, caption_rect in enumerate(captions):
            prev_cap_bottom = 0 if idx == 0 else captions[idx - 1].y1 + params["caption_gap"]
            curr_cap_top = caption_rect.y0
            band_rect = fitz.Rect(page.rect.x0, prev_cap_bottom, page.rect.x1, curr_cap_top)

            band_items = [
                item for item in shape_items if item["rect"].intersects(band_rect) or band_rect.contains(item["rect"])
            ]

            filtered_shapes = []
            for item in band_items:
                rect = item["rect"]
                if item["type"] == "vector":
                    area = rect.width * rect.height
                    if area < params["band_min_area"]:
                        continue
                    aspect = rect.width / rect.height if rect.height > 0 else 0
                    if aspect < params["band_aspect_low"] or aspect > params["band_aspect_high"]:
                        continue
                filtered_shapes.append(rect)

            if not filtered_shapes:
                continue

            figure_box = merge_rectangles(filtered_shapes)
            expanded = expand_with_labels_and_shapes(figure_box, text_blocks, all_vector_rects, captions)
            expanded.y1 = min(expanded.y1, curr_cap_top - params["caption_gap"])
            expanded = clamp_rect_to_page(expanded, page.rect)

            if expanded.width <= 0 or expanded.height <= 0:
                continue

            page_figure_index += 1
            figures.append(
                {
                    "page": page_index + 1,
                    "figure_index": page_figure_index,
                    "caption": caption_text_by_y0.get(caption_rect.y0, ""),
                    "rect": [expanded.x0, expanded.y0, expanded.x1, expanded.y1],
                }
            )

    return figures


def extract_figures_from_path(pdf_path: Path) -> List[dict]:
    doc = fitz.open(pdf_path)
    try:
        return extract_figures_from_document(doc)
    finally:
        doc.close()


def extract_figures(pdf_bytes: bytes) -> List[dict]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        return extract_figures_from_document(doc)
    finally:
        doc.close()


def rect_to_bbox(rect: fitz.Rect) -> dict:
    return {
        "x0": rect.x0,
        "y0": rect.y0,
        "x1": rect.x1,
        "y1": rect.y1,
        "width": rect.width,
        "height": rect.height,
        "unit": "pdf_points",
    }


def format_coord(value: float) -> str:
    return f"{value:.2f}"


def insert_wrapped_text(page: fitz.Page, rect: fitz.Rect, text: str, fontsize: float = 10) -> None:
    page.insert_textbox(
        rect,
        text,
        fontsize=fontsize,
        fontname="helv",
        color=(0.12, 0.12, 0.12),
        lineheight=1.2,
    )


def add_coordinates_report_page(
    report: fitz.Document,
    item: dict,
    source_name: str = "",
    image_bytes: bytes | None = None,
    image_omitted: bool = False,
) -> None:
    bbox = item["bbox"]
    page = report.new_page(width=595, height=842)
    page.insert_text((50, 55), "Figure Coordinates Report", fontsize=18, fontname="helv")
    if source_name:
        page.insert_text((50, 78), f"Source PDF: {source_name}", fontsize=10, fontname="helv")
    page.insert_text(
        (50, 100 if source_name else 85),
        f"Page {item['page']} | Figure {item['figure_index']}",
        fontsize=12,
        fontname="helv",
    )

    coord_text = (
        "Final bounding box coordinates in PDF points, measured from the top-left of the page.\n"
        f"x0: {format_coord(bbox['x0'])}\n"
        f"y0: {format_coord(bbox['y0'])}\n"
        f"x1: {format_coord(bbox['x1'])}\n"
        f"y1: {format_coord(bbox['y1'])}\n"
        f"width: {format_coord(bbox['width'])}\n"
        f"height: {format_coord(bbox['height'])}"
    )
    insert_wrapped_text(page, fitz.Rect(50, 130 if source_name else 115, 545, 240), coord_text, fontsize=10)

    caption = item.get("caption") or "No caption detected."
    insert_wrapped_text(page, fitz.Rect(50, 250, 545, 330), f"Caption: {caption}", fontsize=9)

    image_rect = fitz.Rect(50, 355, 545, 790)
    page.draw_rect(image_rect, color=(0.75, 0.75, 0.75), width=0.5)
    if image_bytes:
        page.insert_image(image_rect, stream=image_bytes, keep_proportion=True)
    elif image_omitted:
        insert_wrapped_text(
            page,
            fitz.Rect(70, 535, 525, 610),
            "Preview omitted to keep the report small. The full PNG crop is included in the figures folder.",
            fontsize=10,
        )


def build_coordinates_pdf(report_items: List[dict], source_name: str = "") -> bytes:
    report = fitz.open()
    try:
        if not report_items:
            page = report.new_page(width=595, height=842)
            page.insert_text((50, 70), "Figure Coordinates Report", fontsize=18, fontname="helv")
            insert_wrapped_text(
                page,
                fitz.Rect(50, 105, 545, 160),
                "No figures were detected in this PDF.",
                fontsize=11,
            )
            return report.tobytes()

        for item in report_items:
            add_coordinates_report_page(report, item, source_name, item.get("image_bytes"))

        return report.tobytes()
    finally:
        report.close()


def write_empty_coordinates_report(report_path: Path, source_name: str = "") -> None:
    report = fitz.open()
    try:
        page = report.new_page(width=595, height=842)
        page.insert_text((50, 70), "Figure Coordinates Report", fontsize=18, fontname="helv")
        if source_name:
            page.insert_text((50, 95), f"Source PDF: {source_name}", fontsize=10, fontname="helv")
        insert_wrapped_text(
            page,
            fitz.Rect(50, 125, 545, 180),
            "No figures were detected in this PDF.",
            fontsize=11,
        )
        report.save(report_path)
    finally:
        report.close()


def build_zip(pdf_bytes: bytes, figures: List[dict], source_name: str = "") -> bytes:
    source_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        buffer = io.BytesIO()
        manifest = []
        report_items = []

        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for figure in figures:
                page = source_doc[figure["page"] - 1]
                rect = fitz.Rect(figure["rect"])
                pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=rect, alpha=False)
                image_bytes = pixmap.tobytes("png")
                image_name = (
                    f"figures/page_{figure['page']:03d}_fig_{figure['figure_index']:02d}.png"
                )
                archive.writestr(image_name, image_bytes)
                bbox = rect_to_bbox(rect)
                manifest.append(
                    {
                        "page": figure["page"],
                        "figure_index": figure["figure_index"],
                        "caption": figure["caption"],
                        "rect": figure["rect"],
                        "bbox": bbox,
                        "crop_path": image_name,
                    }
                )
                report_items.append(
                    {
                        "page": figure["page"],
                        "figure_index": figure["figure_index"],
                        "caption": figure["caption"],
                        "bbox": bbox,
                        "image_bytes": image_bytes,
                    }
                )

            archive.writestr("figures.json", json.dumps(manifest, ensure_ascii=True, indent=2))
            archive.writestr("figure_coordinates.pdf", build_coordinates_pdf(report_items, source_name))

        return buffer.getvalue()
    finally:
        source_doc.close()


def build_zip_to_path(pdf_path: Path, zip_path: Path, figures: List[dict], source_name: str = "") -> None:
    source_doc = fitz.open(pdf_path)
    report = fitz.open()
    report_path = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name)
    try:
        manifest = []

        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for figure_index, figure in enumerate(figures):
                page = source_doc[figure["page"] - 1]
                rect = fitz.Rect(figure["rect"])
                pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=rect, alpha=False)
                image_bytes = pixmap.tobytes("png")
                image_name = f"figures/page_{figure['page']:03d}_fig_{figure['figure_index']:02d}.png"
                archive.writestr(image_name, image_bytes)

                bbox = rect_to_bbox(rect)
                manifest_item = {
                    "page": figure["page"],
                    "figure_index": figure["figure_index"],
                    "caption": figure["caption"],
                    "rect": figure["rect"],
                    "bbox": bbox,
                    "crop_path": image_name,
                }
                manifest.append(manifest_item)

                add_coordinates_report_page(
                    report,
                    manifest_item,
                    source_name,
                    image_bytes if figure_index < REPORT_IMAGE_PREVIEW_LIMIT else None,
                    image_omitted=figure_index >= REPORT_IMAGE_PREVIEW_LIMIT,
                )
                del pixmap, image_bytes

            archive.writestr("figures.json", json.dumps(manifest, ensure_ascii=True, indent=2))

            if figures:
                report.save(report_path)
            else:
                report.close()
                report = None
                write_empty_coordinates_report(report_path, source_name)
            archive.write(report_path, "figure_coordinates.pdf")
    finally:
        if report is not None:
            report.close()
        source_doc.close()
        safe_unlink(report_path)


def safe_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass


async def save_upload_to_temp(file: UploadFile) -> Path:
    suffix = Path(file.filename or "").suffix or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = Path(tmp.name)
        while True:
            chunk = await file.read(UPLOAD_CHUNK_SIZE)
            if not chunk:
                break
            tmp.write(chunk)
    return tmp_path


def cleanup_paths(*paths: Path) -> None:
    for path in paths:
        safe_unlink(path)


@app.get("/health")
def healthcheck():
    return {"status": "ok"}


@app.get("/")
def index():
    if not FRONTEND_FILE.exists():
        raise HTTPException(status_code=404, detail="Frontend file not found.")
    return FileResponse(FRONTEND_FILE)


@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    pdf_path = await save_upload_to_temp(file)
    zip_path = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".zip").name)
    safe_unlink(zip_path)
    if pdf_path.stat().st_size == 0:
        safe_unlink(pdf_path)
        raise HTTPException(status_code=400, detail="Uploaded PDF was empty.")

    try:
        figures = extract_figures_from_path(pdf_path)
        build_zip_to_path(pdf_path, zip_path, figures, file.filename)
    except HTTPException:
        cleanup_paths(pdf_path, zip_path)
        raise
    except Exception as exc:
        cleanup_paths(pdf_path, zip_path)
        raise HTTPException(status_code=500, detail=f"Extraction failed: {exc}") from exc

    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename="figures.zip",
        background=BackgroundTask(cleanup_paths, pdf_path, zip_path),
    )
