"""Node picker dialog for the Pipelines tab (V1.45.1).

A roomy modal palette for the **Add** button: a scrollable list of every node
type on the left and, on the right, a lengthy plain-language explanation of what
the selected operation does plus a large live *raw → processed* thumbnail pair
(a synthetic microscopy image run through the plugin at its default params, with
an arrow between them). Double-click a row — or pick one and press **Add node**
— to drop it on the board.

The synthetic "universal" image is deliberately busy so every plugin shows a
visible effect: cell-like blobs across a range of sizes, large diffuse debris,
fine textured background, spatial illumination heterogeneity (vignette + a
diagonal gradient), additive read noise, and scattered hot pixels. The preview
runs the *real* enhancement plugin (``PluginBase`` registry) on that image, so
it always matches what the node will actually do; execution is wrapped
defensively so anything that raises falls back to "preview unavailable".
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from nd2studios.core.plugin_registry import PluginBase
from nd2studios.core.settings import Settings
from nd2studios.pipeline_graph.model import NodeCategory, NodeRole, Stage
from nd2studios.pipeline_graph.registry_adapter import (
    NodeSpec,
    default_params_for,
    param_specs_for,
    plugin_name_for_op_key,
)
from nd2studios.widgets.icon_button import scaled

# Generation resolution (image px) of the synthetic field, and the on-screen
# thumbnail edge (base 96-DPI px). The two are decoupled so the preview can be
# large while the source stays a sensible size for the heavier denoisers.
_RAW_RES = 240
_THUMB = 210

# Order + labels for the Add-dialog category tabs (V1.45 merge).
_CATEGORY_ORDER = [
    NodeCategory.PROCESSING, NodeCategory.ANALYSIS, NodeCategory.RESULTS,
    NodeCategory.LOGIC, NodeCategory.SPECIAL, NodeCategory.CHECKPOINT,
]
_CATEGORY_LABEL = {
    NodeCategory.PROCESSING: "Processing",
    NodeCategory.ANALYSIS: "Analysis",
    NodeCategory.RESULTS: "Results",
    NodeCategory.LOGIC: "Logic",
    NodeCategory.SPECIAL: "Special",
    NodeCategory.CHECKPOINT: "Checkpoint",  # V1.53 freeze/cache node
}


@lru_cache(maxsize=1)
def _synthetic_raw(res: int) -> np.ndarray:
    """A reproducible, content-rich microscopy-like (H, W) uint16 image.

    Cached so reopening the dialog (and re-selecting rows) is instant.
    """
    rng = np.random.default_rng(7)
    yy, xx = np.mgrid[0:res, 0:res].astype(np.float32)
    img = np.zeros((res, res), np.float32)

    # Many cell-like blobs across a range of sizes/intensities — exercises
    # blur / median / DoG / top-hat / local-contrast / gradient.
    for _ in range(60):
        cy = rng.uniform(0.05, 0.95) * res
        cx = rng.uniform(0.05, 0.95) * res
        r = rng.uniform(0.012, 0.05) * res
        img += rng.uniform(0.3, 1.0) * np.exp(
            -(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r)))

    # A few large diffuse debris blobs — background-subtract / blob-subtract.
    for _ in range(3):
        cy = rng.uniform(0.1, 0.9) * res
        cx = rng.uniform(0.1, 0.9) * res
        r = rng.uniform(0.09, 0.16) * res
        img += rng.uniform(0.6, 1.0) * np.exp(
            -(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r)))

    # Fine textured background — denoisers / unsharp / bilateral.
    img += 0.08 * rng.standard_normal((res, res)).astype(np.float32)

    # Spatial illumination heterogeneity: a centred vignette times a diagonal
    # gradient, applied multiplicatively plus a small additive offset — the
    # target of flat-field / background / local-contrast / CLAHE.
    cx0, cy0 = res * 0.5, res * 0.45
    vignette = np.exp(-(((xx - cx0) ** 2 + (yy - cy0) ** 2)
                        / (2 * (0.6 * res) ** 2)))
    gradient = 0.5 + 0.5 * ((xx + yy) / (2 * res))
    illum = (0.35 + 0.65 * vignette) * gradient
    img = img * illum + 0.12 * illum

    # Additive read noise + scattered hot pixels (salt) — median / blob-subtract.
    img += 0.04 * rng.standard_normal((res, res)).astype(np.float32)
    n_hot = int(res * res * 0.0008)
    ys = rng.integers(0, res, n_hot)
    xs = rng.integers(0, res, n_hot)
    img[ys, xs] += rng.uniform(0.8, 1.5, n_hot)

    img = np.clip(img, 0.0, None)
    img = img / (img.max() + 1e-9)
    return (img * 60000.0).astype(np.uint16)


def _to_pixmap(arr2d: np.ndarray, edge: int) -> QPixmap:
    """Contrast-stretch a 2-D array to 8-bit grayscale and scale to ``edge``."""
    a = np.asarray(arr2d).astype(np.float32)
    if a.ndim != 2:
        a = a.reshape(a.shape[-2], a.shape[-1]) if a.ndim >= 2 else a.ravel()[None, :]
    lo, hi = float(a.min()), float(a.max())
    a = (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)
    u8 = np.ascontiguousarray((a * 255).astype(np.uint8))
    h, w = u8.shape
    img = QImage(u8.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
    return QPixmap.fromImage(img).scaled(
        edge, edge, Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


def _process(spec: NodeSpec, raw: np.ndarray) -> Optional[np.ndarray]:
    """Run ``spec``'s enhancement plugin on ``raw`` (H, W); ``None`` if N/A."""
    if spec.role is not NodeRole.ACTION:
        return None
    cls = PluginBase.get_plugin("enhancement", plugin_name_for_op_key(spec.op_key))
    if cls is None:
        return None
    params = default_params_for(spec.op_key)
    for dtype in (raw.dtype, np.uint8):  # some ops (e.g. large median) want 8-bit
        try:
            out = cls().execute(raw.astype(dtype)[None, ...], params, progress_cb=None)
            return np.asarray(out)[0]
        except Exception:  # noqa: BLE001 — preview must never crash the picker
            continue
    return None


# Lengthy, plain-language explanations keyed by plugin name (``cls.name``). Two
# synthetic keys cover the source/sink nodes. Anything missing falls back to the
# plugin's own one-line ``description``.
_DETAILS: Dict[str, str] = {
    "Normalize":
        "Rescales intensities so a chosen low/high percentile of pixels map to "
        "black and white and everything between is stretched across the full "
        "range. This standardises brightness and contrast across frames and "
        "channels so later steps behave consistently regardless of the original "
        "exposure; pixels beyond the percentiles are clipped, which also tames a "
        "handful of very bright outliers.",
    "CLAHE":
        "Contrast Limited Adaptive Histogram Equalization. The image is split "
        "into a grid of tiles and the histogram is equalized within each one, so "
        "faint local detail in both dark and bright regions becomes visible "
        "without any region dominating. The clip limit caps how much contrast a "
        "tile may gain, which stops flat noisy areas from being amplified into "
        "blotches.",
    "Gaussian Blur":
        "Convolves each frame with a Gaussian kernel, smoothly averaging every "
        "pixel with its neighbours. It suppresses high-frequency noise and small "
        "specks at the cost of fine detail and edge sharpness — larger sigma "
        "blurs more strongly over a wider neighbourhood. Frequently used to "
        "denoise gently or to build a smooth background estimate for other steps.",
    "Median Filter":
        "Replaces each pixel with the median of a square neighbourhood. Unlike a "
        "mean or Gaussian blur it erases salt-and-pepper noise and isolated hot "
        "pixels while keeping edges crisp, because a lone outlier never survives "
        "the median. Larger kernels remove bigger specks but begin to erode "
        "genuine small features.",
    "Background Subtract":
        "Estimates the slowly-varying background — by a rolling-ball "
        "morphological opening or a large Gaussian blur — and subtracts it, "
        "flattening uneven illumination and removing diffuse haze so objects sit "
        "on a near-zero baseline. The radius sets the size of structures treated "
        "as background: anything larger is removed, anything smaller is kept.",
    "Gamma Correction":
        "Applies a power-law transform to intensities (output = input^gamma). "
        "A gamma below 1 brightens midtones and reveals detail in dark regions; "
        "above 1 it darkens midtones and emphasises bright structures. It is a "
        "non-linear contrast control that leaves pure black and white fixed "
        "while reshaping everything in between.",
    "Bleach Correction":
        "Compensates for photobleaching by rescaling each frame so its mean "
        "intensity matches a reference, counteracting the gradual fade of "
        "fluorescence over a time-lapse. Brightness then stays comparable from "
        "the first frame to the last — important whenever intensity should "
        "reflect biology rather than dye loss.",
    "Temporal Fold Correction":
        "Detects frames whose mean intensity strays from a rolling average of "
        "their neighbours and clamps them back toward it. This repairs sporadic "
        "acquisition glitches — a flash, a dropped or suddenly-dim frame — that "
        "would otherwise flicker in a time-lapse, while leaving the well-behaved "
        "frames around them untouched.",
    "Spatial Flatness":
        "Flattens large-scale illumination variation across the field of view, "
        "correcting vignetting and an uneven lamp or laser profile so the "
        "background reads uniformly bright from centre to edge. It divides out a "
        "smooth estimate of the lighting, leaving object intensities comparable "
        "no matter where they sit in the frame.",
    "Top-Hat":
        "A morphological white top-hat: it subtracts a morphological opening "
        "from the image, isolating bright features smaller than the structuring "
        "element while discarding larger background swells. Ideal for pulling "
        "small puncta or spots off an uneven or gradually varying background; "
        "the kernel size sets the largest feature kept.",
    "Difference of Gaussians":
        "Subtracts a more-blurred copy of the image from a less-blurred one, "
        "forming a band-pass filter that keeps features whose size lies between "
        "the two blur scales. It enhances blobs and edges of a chosen size while "
        "suppressing both fine noise and broad background — the classic blob / "
        "spot detector front-end.",
    "Unsharp Mask":
        "Sharpens by adding back a scaled copy of the difference between the "
        "image and a blurred version of itself (output = original + amount·"
        "(original − blurred)). This accentuates edges and fine detail; the "
        "amount sets sharpening strength and the radius the scale of detail "
        "enhanced. Too much exaggerates noise and rings edges with halos.",
    "Bilateral Denoise":
        "Edge-preserving smoothing that averages neighbouring pixels weighted by "
        "both spatial distance and intensity similarity. Flat regions are "
        "smoothed to remove noise, but pixels across a strong edge are never "
        "mixed together, so borders stay sharp — unlike a plain Gaussian blur, "
        "which softens everything indiscriminately.",
    "Morphological Gradient":
        "The difference between a dilation and an erosion of the image, which is "
        "large only where intensity changes rapidly — i.e. at object boundaries. "
        "The result draws cell outlines and edges as bright ridges over a dark "
        "interior, useful for visualising structure or as input to a "
        "segmentation step.",
    "Local Contrast":
        "Divides the image by a heavily blurred, large-scale version of itself, "
        "normalising out slow brightness variation and boosting local contrast "
        "everywhere. Faint structures in dim regions become as visible as those "
        "in bright regions; the blur scale sets the size above which variation "
        "is treated as illumination to remove.",
    "Blob Subtract":
        "Detects compact bright blobs — dust, debris, hot pixels — and removes "
        "them by replacing each with an estimate of its surroundings. This "
        "cleans acquisition artefacts that would otherwise be mistaken for real "
        "objects, while leaving genuine connected structures intact; the "
        "thresholds control how bright and large a blob must be to be removed.",
    "NLM Denoise":
        "Non-Local Means denoising averages each patch with other similar-"
        "looking patches from across the whole image, not just nearby pixels. "
        "Because repeated texture reinforces the true signal while noise cancels "
        "out, it removes noise while preserving fine detail and edges remarkably "
        "well — at a noticeably higher computational cost than local filters.",
    "Wavelet Denoise":
        "Transforms the image into a multi-scale wavelet representation, shrinks "
        "the small coefficients that mostly carry noise, and inverts the "
        "transform. It removes noise across several scales at once while "
        "preserving edges better than a uniform blur, since sharp features live "
        "in the large coefficients that are kept.",
    "TV Denoise":
        "Total-Variation denoising finds the image closest to the input that "
        "also has low total variation (the summed magnitude of intensity "
        "gradients). The result strips noise while keeping sharp edges and "
        "yielding clean, piecewise-smooth regions — at the cost of a slightly "
        "'cartoon-like' flattening of gentle gradients; the weight trades "
        "denoising strength against fidelity to the original.",
    "__input__":
        "The image channels loaded from Import — the source every pipeline "
        "starts from. Each channel is a (T, H, W) stack; wire this node's output "
        "into operations to build a recipe. There is exactly one Input node per "
        "Processing graph.",
    "__output__":
        "Marks a processed result and bridges it onward to Analysis and Export. "
        "The unique chain of operations from the Input up to this node defines "
        "its recipe; an Output may be fed by any branch, and several Outputs can "
        "capture different branches of the same graph.",
    # ── Analysis pipelines (keyed by AnalysisPipeline.name) ─────────────────
    "Nuclei Segmentation":
        "Segments nuclei with the Cellpose 3 deep-learning model, producing one "
        "labelled mask per detected nucleus. It handles touching and "
        "irregularly-shaped objects far better than a plain threshold, at the "
        "cost of a heavier dependency and longer runtime. Cellpose is an "
        "optional install — if it is missing, the node reports a friendly error "
        "rather than crashing.",
    "Histogram Threshold Segmenter":
        "Separates foreground from background by intensity threshold (Otsu, "
        "Triangle, Minimum, Multi-Otsu, percentile or a fixed/relative cut), "
        "then labels the connected components. Optional spatial constraints and "
        "size filtering clean up the result. A fast, classical segmenter for "
        "well-separated, reasonably bright objects.",
    "Bright / Dark Spots":
        "A GA3-style scale-space spot detector: it finds compact bright (or "
        "dark) blobs across a range of sizes and reports each spot's diameter, "
        "contrast, circularity and polarity. Ideal for puncta, foci, or "
        "vesicle-like features rather than large filled regions.",
    "Tear / Dark Region Detection":
        "Classical detector for dark, homogeneous regions (e.g. tears or "
        "voids): it scores areas by intensity, variance and entropy and labels "
        "those that read as uniformly dark, reporting homogeneity, solidity and "
        "eccentricity per region.",
    "Mask Analysis":
        "Measures user-defined regions — hand-drawn shapes and auto-tiled "
        "strips — rather than auto-segmenting. The drawing tools that create "
        "those shapes live on the Analysis page; here the node measures whatever "
        "masks the session already holds.",
    "__analysis_input__":
        "The processed image channels bridged from Processing — the source the "
        "analysis pipelines run on. Wire it into a pipeline node to segment or "
        "measure. There is one input per Analysis graph.",
    "__analysis_output__":
        "Marks an analysis result (label masks + measurements) and bridges it "
        "onward to Results / Export. Fed by a single pipeline node.",
    # ── Results ops ─────────────────────────────────────────────────────────
    "Compute Measurements":
        "Computes per-object measurements for a chosen analysis result: area in "
        "pixels and µm², centroids (pixel, µm and stage-absolute µm), perimeter, "
        "circularity, eccentricity, solidity, bounding box and per-channel "
        "mean/std intensity — the rows shown in the results table. Pick which "
        "committed analysis result to measure in the parameters.",
    "Summary":
        "Aggregate statistics for a chosen analysis result: total object count, "
        "number of frames with objects, and mean / standard-deviation area. A "
        "quick headline read on a segmentation before exporting the full table.",
    "__results_input__":
        "The analysis results (label masks + measurements) bridged from Analysis "
        "— the source the results operations measure and overlay.",
    "__results_output__":
        "Marks an exportable result (the measurements table) — surfaced in the "
        "Export page's source selector.",
}


def _describe(spec: NodeSpec) -> str:
    """Lengthy explanation: a curated paragraph plus the tunable parameters."""
    detail = _DETAILS.get(spec.title)
    if detail is None:
        if spec.stage is Stage.ANALYSIS:
            in_key, out_key = "__analysis_input__", "__analysis_output__"
        elif spec.stage is Stage.RESULTS:
            in_key, out_key = "__results_input__", "__results_output__"
        else:
            in_key, out_key = "__input__", "__output__"
        if spec.role is NodeRole.OUTPUT:
            detail = _DETAILS[out_key]
        elif spec.role is NodeRole.INPUT:
            detail = _DETAILS[in_key]
        else:
            detail = (spec.description or "").strip() or (
                "Applies this operation to every frame of each channel.")
    specs = param_specs_for(spec.op_key)
    if specs:
        lines = []
        for s in specs:
            tip = (s.tooltip or "").strip()
            lines.append(f"   •  {s.label}: {tip}" if tip else f"   •  {s.label}")
        detail += "\n\nParameters:\n" + "\n".join(lines)
    return detail


class _Thumb(QLabel):
    """Fixed-size framed image slot."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("addNodeThumb")
        self.setFixedSize(scaled(_THUMB), scaled(_THUMB))
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setText("—")


class AddNodeDialog(QDialog):
    """Pick a node type to add. ``selected_spec()`` is valid after ``Accepted``."""

    def __init__(self, specs: List[NodeSpec], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add node")
        self.setObjectName("addNodeDialog")
        self.setModal(True)
        self.setMinimumSize(scaled(900), scaled(640))
        self._specs = list(specs)
        self._selected: Optional[NodeSpec] = None
        self._proc_cache: Dict[str, Optional[np.ndarray]] = {}
        self._raw = _synthetic_raw(_RAW_RES)
        self._lists: List[QListWidget] = []
        self._build_ui()
        # Select the first row of the first tab so the detail pane is populated.
        if self._tabs.count():
            first = self._tabs.widget(0)
            if isinstance(first, QListWidget) and first.count():
                first.setCurrentRow(0)

    # ── UI ──────────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(12)

        row = QHBoxLayout()
        row.setSpacing(14)

        # Left — the catalog, grouped into per-category tabs (V1.45 merge). Each
        # item carries its GLOBAL index into ``self._specs`` so the shared detail
        # pane works regardless of which tab it lives in.
        groups: Dict[NodeCategory, List[int]] = {}
        for i, spec in enumerate(self._specs):
            groups.setdefault(spec.effective_category(), []).append(i)

        self._tabs = QTabWidget()
        self._tabs.setObjectName("addNodeTabs")
        self._tabs.setFixedWidth(scaled(230))
        for cat in _CATEGORY_ORDER:
            idxs = groups.get(cat)
            if not idxs:
                continue
            lst = QListWidget()
            lst.setObjectName("addNodeList")
            for gi in idxs:
                spec = self._specs[gi]
                item = QListWidgetItem(spec.title)
                item.setData(Qt.ItemDataRole.UserRole, gi)
                item.setToolTip(spec.description or spec.title)
                lst.addItem(item)
            lst.currentItemChanged.connect(self._on_item_changed)
            lst.itemDoubleClicked.connect(lambda _i: self.accept())
            self._lists.append(lst)
            self._tabs.addTab(lst, _CATEGORY_LABEL.get(cat, cat.value.title()))
        self._tabs.currentChanged.connect(self._on_tab_changed)
        row.addWidget(self._tabs)

        # Right — detail pane.
        detail = QVBoxLayout()
        detail.setSpacing(10)

        self._title = QLabel("")
        self._title.setObjectName("addNodeTitle")
        self._title.setStyleSheet(f"color:{Settings.ACCENT_CYAN}; font:bold 15pt;")
        detail.addWidget(self._title)

        # Scroll the (lengthy) description so it never crowds out the preview.
        self._desc = QLabel("")
        self._desc.setWordWrap(True)
        self._desc.setObjectName("addNodeDesc")
        self._desc.setTextFormat(Qt.TextFormat.PlainText)
        self._desc.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self._desc.setStyleSheet(
            f"color:{Settings.FG_PRIMARY}; font:10.5pt; line-height:140%;")
        desc_scroll = QScrollArea()
        desc_scroll.setObjectName("addNodeDescScroll")
        desc_scroll.setWidgetResizable(True)
        desc_scroll.setFrameShape(QFrame.Shape.NoFrame)
        desc_scroll.setWidget(self._desc)
        desc_scroll.setMinimumWidth(scaled(440))
        detail.addWidget(desc_scroll, stretch=1)

        # raw → processed thumbnails.
        prev = QFrame()
        prev.setObjectName("addNodePreview")
        prev_l = QHBoxLayout(prev)
        prev_l.setContentsMargins(12, 12, 12, 12)
        prev_l.setSpacing(12)
        self._raw_thumb = self._captioned("Raw (example data)")
        self._arrow = QLabel("→")
        self._arrow.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._arrow.setStyleSheet(
            f"color:{Settings.ACCENT_CYAN}; font:bold 28pt; background:transparent;")
        self._proc_thumb = self._captioned("Processed result")
        prev_l.addStretch(1)
        prev_l.addLayout(self._raw_thumb[0])
        prev_l.addWidget(self._arrow)
        prev_l.addLayout(self._proc_thumb[0])
        prev_l.addStretch(1)
        detail.addWidget(prev)

        row.addLayout(detail, stretch=1)
        outer.addLayout(row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Add node")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

    def _captioned(self, caption: str):
        """Return ``(layout, thumb_label)`` — a thumbnail with a caption below."""
        col = QVBoxLayout()
        col.setSpacing(5)
        thumb = _Thumb()
        cap = QLabel(caption)
        cap.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cap.setStyleSheet(f"color:{Settings.FG_SECONDARY}; font:8.5pt;")
        col.addWidget(thumb, alignment=Qt.AlignmentFlag.AlignCenter)
        col.addWidget(cap)
        return col, thumb

    # ── selection / preview ──────────────────────────────────────────────
    def _on_item_changed(self, current, _previous) -> None:
        """A list selection changed — resolve its global spec index."""
        if current is None:
            return
        idx = current.data(Qt.ItemDataRole.UserRole)
        if idx is not None:
            self._on_row_changed(int(idx))

    def _on_tab_changed(self, _index: int) -> None:
        """Switching tabs re-populates the detail pane from the new tab's list."""
        lst = self._tabs.currentWidget()
        if not isinstance(lst, QListWidget):
            return
        if lst.currentItem() is None and lst.count():
            lst.setCurrentRow(0)
        else:
            self._on_item_changed(lst.currentItem(), None)

    def _on_row_changed(self, row: int) -> None:
        if not (0 <= row < len(self._specs)):
            return
        spec = self._specs[row]
        self._selected = spec
        self._title.setText(spec.title)
        self._desc.setText(_describe(spec))

        edge = scaled(_THUMB)
        self._raw_thumb[1].setPixmap(_to_pixmap(self._raw, edge))

        proc = self._processed_for(spec)
        if proc is None:
            self._arrow.setVisible(spec.role is NodeRole.ACTION)
            if (spec.stage in (Stage.ANALYSIS, Stage.RESULTS)
                    and spec.role is NodeRole.ACTION):
                note = "overlay + table shown\nlive in the viewer"
            elif spec.role is NodeRole.ACTION:
                note = "preview unavailable"
            else:
                note = "passes through unchanged"
            self._proc_thumb[1].setText(note)
            self._proc_thumb[1].setPixmap(QPixmap())
        else:
            self._arrow.setVisible(True)
            self._proc_thumb[1].setPixmap(_to_pixmap(proc, edge))

    def _processed_for(self, spec: NodeSpec) -> Optional[np.ndarray]:
        if spec.op_key not in self._proc_cache:
            self._proc_cache[spec.op_key] = _process(spec, self._raw)
        return self._proc_cache[spec.op_key]

    # ── result ────────────────────────────────────────────────────────────
    def selected_spec(self) -> Optional[NodeSpec]:
        return self._selected
