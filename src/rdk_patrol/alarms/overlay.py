from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from rdk_patrol.contracts import AlarmCandidate


RGBColor = Tuple[int, int, int]


def _font_candidates() -> Iterable[Path]:
    configured = os.environ.get("RDK_PATROL_CJK_FONT")
    if configured:
        yield Path(configured)
    for value in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
    ):
        yield Path(value)


def find_cjk_font() -> Optional[Path]:
    """Return the first installed CJK-capable font, if one is available."""

    for candidate in _font_candidates():
        if candidate.is_file():
            return candidate
    return None


def _load_font(size: int, font_path: Optional[Path]) -> ImageFont.ImageFont:
    selected = font_path or find_cjk_font()
    if selected is not None:
        try:
            return ImageFont.truetype(str(selected), size=size)
        except OSError:
            pass
    try:
        # DejaVu is commonly bundled with Pillow/Linux. It keeps Unicode text
        # rendering safe (tofu glyphs if needed) when no CJK font was deployed.
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        pass
    return ImageFont.load_default()


def _coerce_box(item: Any) -> Optional[Tuple[float, float, float, float]]:
    value = item
    if isinstance(item, Mapping):
        value = item.get("box", item.get("box_xyxy"))
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    if len(value) != 4:
        return None
    try:
        return tuple(float(part) for part in value)  # type: ignore[return-value]
    except (TypeError, ValueError):
        return None


def _coerce_points(item: Any) -> Optional[List[Tuple[float, float]]]:
    value = item
    if isinstance(item, Mapping):
        value = item.get("points", item.get("polygon"))
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    result: List[Tuple[float, float]] = []
    try:
        for point in value:
            if not isinstance(point, Sequence) or len(point) != 2:
                return None
            result.append((float(point[0]), float(point[1])))
    except (TypeError, ValueError):
        return None
    return result if len(result) >= 2 else None


def _item_label(item: Any, fallback: str) -> str:
    if isinstance(item, Mapping):
        label = item.get("label")
        if label is not None:
            return str(label)
    return fallback


class PillowEvidenceRenderer:
    """Render human-readable evidence without mutating the source BGR frame.

    Pillow is deliberately used for the text layer because OpenCV's built-in
    fonts do not render Chinese alarm names reliably on the board.
    """

    def __init__(
        self,
        font_path: Optional[Path] = None,
        font_size: int = 26,
        line_width: int = 3,
    ) -> None:
        self.font_path = font_path
        self.font_size = max(12, int(font_size))
        self.line_width = max(1, int(line_width))

    def render(
        self,
        candidate: AlarmCandidate,
        beijing_time: str,
    ) -> Image.Image:
        frame = np.asarray(candidate.frame_bgr)
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
            raise ValueError("alarm frame_bgr must be a non-empty HxWx3 image")
        rgb = np.ascontiguousarray(frame[:, :, ::-1])
        image = Image.fromarray(rgb, mode="RGB")
        draw = ImageDraw.Draw(image, mode="RGBA")
        font = _load_font(self.font_size, self.font_path)
        small_font = _load_font(max(12, self.font_size - 4), self.font_path)

        evidence = candidate.evidence or {}
        self._draw_polygons(draw, evidence, small_font)
        self._draw_boxes(draw, evidence, small_font)
        lines = self._headline(candidate, beijing_time)
        lines.extend(self._evidence_lines(evidence))
        self._draw_text_panel(draw, image.size, lines, font)
        return image

    def _draw_boxes(
        self,
        draw: ImageDraw.ImageDraw,
        evidence: Mapping[str, Any],
        font: ImageFont.ImageFont,
    ) -> None:
        boxes = evidence.get("boxes", [])
        if not isinstance(boxes, Sequence) or isinstance(boxes, (str, bytes)):
            return
        for index, item in enumerate(boxes):
            box = _coerce_box(item)
            if box is None:
                continue
            label = _item_label(item, "目标{}".format(index + 1))
            x1, y1, x2, y2 = box
            color: RGBColor = (255, 64, 64)
            draw.rectangle(
                (x1, y1, x2, y2),
                outline=color + (255,),
                width=self.line_width,
            )
            if label:
                self._draw_label(draw, (x1, max(0.0, y1 - self.font_size)), label, font, color)

    def _draw_polygons(
        self,
        draw: ImageDraw.ImageDraw,
        evidence: Mapping[str, Any],
        font: ImageFont.ImageFont,
    ) -> None:
        raw = evidence.get("polygons")
        if raw is None and evidence.get("roi") is not None:
            raw = [evidence.get("roi")]
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            return
        for index, item in enumerate(raw):
            points = _coerce_points(item)
            if points is None:
                continue
            label = _item_label(item, "敏感区域")
            draw.line(
                points + [points[0]],
                fill=(255, 190, 0, 255),
                width=self.line_width,
                joint="curve",
            )
            self._draw_label(draw, points[0], label, font, (255, 190, 0))

    @staticmethod
    def _draw_label(
        draw: ImageDraw.ImageDraw,
        position: Tuple[float, float],
        label: str,
        font: ImageFont.ImageFont,
        color: RGBColor,
    ) -> None:
        x, y = position
        bbox = draw.textbbox((x, y), label, font=font)
        draw.rectangle(bbox, fill=(0, 0, 0, 190))
        draw.text((x, y), label, font=font, fill=color + (255,))

    @staticmethod
    def _headline(candidate: AlarmCandidate, beijing_time: str) -> List[str]:
        lines = ["事件名称：{}".format(candidate.event_name), "北京时间：{}".format(beijing_time)]
        if candidate.point_name:
            lines.append("点位名称：{}".format(candidate.point_name))
        return lines

    @staticmethod
    def _evidence_lines(evidence: Mapping[str, Any]) -> List[str]:
        explicit = evidence.get("overlay_lines")
        if isinstance(explicit, Sequence) and not isinstance(explicit, (str, bytes)):
            return [str(line) for line in explicit if str(line)]

        lines: List[str] = []
        if "person_count" in evidence:
            lines.append("人数：{}".format(evidence["person_count"]))
        if "confidence" in evidence:
            try:
                lines.append("置信度：{:.1%}".format(float(evidence["confidence"])))
            except (TypeError, ValueError):
                lines.append("置信度：{}".format(evidence["confidence"]))
        if "distance_m" in evidence:
            value = evidence.get("distance_m")
            if value is None:
                lines.append("距离：不可用")
            else:
                try:
                    lines.append("距离：{:.2f} 米".format(float(value)))
                except (TypeError, ValueError):
                    lines.append("距离：不可用")
        elif evidence.get("distance_unavailable"):
            lines.append("距离：不可用")
        if "review_result" in evidence:
            lines.append("MiniMax结论：{}".format(evidence["review_result"]))
        return lines

    def _draw_text_panel(
        self,
        draw: ImageDraw.ImageDraw,
        image_size: Tuple[int, int],
        lines: Sequence[str],
        font: ImageFont.ImageFont,
    ) -> None:
        if not lines:
            return
        width, height = image_size
        padding = 10
        line_height = self.font_size + 8
        panel_height = min(height, padding * 2 + line_height * len(lines))
        draw.rectangle((0, 0, width, panel_height), fill=(0, 0, 0, 175))
        y = padding
        for line in lines:
            draw.text((padding, y), str(line), font=font, fill=(255, 255, 255, 255))
            y += line_height
