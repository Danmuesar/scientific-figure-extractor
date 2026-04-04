import io
import json
import zipfile
from pathlib import Path
from typing import List

import fitz
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response


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


def extract_figures(pdf_bytes: bytes) -> List[dict]:
    figures = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
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
    finally:
        doc.close()

    return figures


def build_zip(pdf_bytes: bytes, figures: List[dict]) -> bytes:
    source_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        buffer = io.BytesIO()
        manifest = []

        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for figure in figures:
                page = source_doc[figure["page"] - 1]
                rect = fitz.Rect(figure["rect"])
                pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=rect, alpha=False)
                image_name = (
                    f"figures/page_{figure['page']:03d}_fig_{figure['figure_index']:02d}.png"
                )
                archive.writestr(image_name, pixmap.tobytes("png"))
                manifest.append(
                    {
                        "page": figure["page"],
                        "figure_index": figure["figure_index"],
                        "caption": figure["caption"],
                        "crop_path": image_name,
                    }
                )

            archive.writestr("figures.json", json.dumps(manifest, ensure_ascii=True, indent=2))

        return buffer.getvalue()
    finally:
        source_doc.close()


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

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Uploaded PDF was empty.")

    try:
        figures = extract_figures(pdf_bytes)
        zip_bytes = build_zip(pdf_bytes, figures)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {exc}") from exc

    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=figures.zip"},
    )
