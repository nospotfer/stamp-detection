"""Box drawing on document images with PIL (no cv2 dependency).

Convention: ground truth in green, predictions in red with score labels.
Boxes are absolute xyxy in the coordinate space of the image being drawn on.
"""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

GT_COLOR = (0, 170, 0)
PRED_COLOR = (220, 30, 30)


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_box(draw: ImageDraw.ImageDraw, box, color, label: str | None,
              width: int, font) -> None:
    x1, y1, x2, y2 = (float(v) for v in box)
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    if label:
        tb = draw.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        ty = y1 - th - 6 if y1 - th - 6 > 0 else y2 + 2
        draw.rectangle([x1, ty, x1 + tw + 6, ty + th + 6], fill=color)
        draw.text((x1 + 3, ty + 2), label, fill=(255, 255, 255), font=font)


def draw_detections(
    image: Image.Image,
    pred_boxes=(),
    pred_scores=(),
    gt_boxes=(),
    line_width: int | None = None,
) -> Image.Image:
    """Return a copy of `image` with GT (green) and predicted (red + score) boxes drawn."""
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    width = line_width or max(2, round(min(canvas.size) / 300))
    font = _font(max(12, round(min(canvas.size) / 50)))
    for box in gt_boxes:
        _draw_box(draw, box, GT_COLOR, "stamp (GT)", width, font)
    for box, score in zip(pred_boxes, pred_scores):
        _draw_box(draw, box, PRED_COLOR, f"stamp {float(score):.2f}", width, font)
    return canvas


def side_by_side(image: Image.Image, pred_boxes, pred_scores, gt_boxes) -> Image.Image:
    """Two panels: left = ground truth, right = predictions."""
    left = draw_detections(image, gt_boxes=gt_boxes)
    right = draw_detections(image, pred_boxes=pred_boxes, pred_scores=pred_scores)
    combo = Image.new("RGB", (left.width + right.width, left.height), (255, 255, 255))
    combo.paste(left, (0, 0))
    combo.paste(right, (left.width, 0))
    return combo
