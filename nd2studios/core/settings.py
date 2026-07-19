"""
Application constants, Dracula palette, page list, sidebar/titlebar config.

All UI dimensions and colors live here so theming and layout tweaks happen
in one place.
"""
from __future__ import annotations
import os as _os


class Settings:
    # Root of the ND2Studios repository (two levels up from this file).
    PROJECT_DIR: str = _os.path.abspath(
        _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..")
    )
    APP_NAME = "ND2Studios"
    APP_VERSION = "1.0.0"
    APP_DESCRIPTION = "ND2 file processing and export — McGhee Lab"

    # Window
    MIN_WIDTH = 1280
    MIN_HEIGHT = 820

    # Sidebar dimensions (PyDracula-style collapsible)
    SIDEBAR_COLLAPSED_WIDTH = 60
    SIDEBAR_EXPANDED_WIDTH = 240
    SIDEBAR_ANIMATION_MS = 280  # InOutQuart easing

    # Top title bar
    TITLE_BAR_HEIGHT = 40
    BOTTOM_BAR_HEIGHT = 28

    # Drop shadow on the app background frame (PyDracula chrome)
    SHADOW_BLUR_RADIUS = 17
    SHADOW_OFFSET_X = 0
    SHADOW_OFFSET_Y = 0
    SHADOW_ALPHA = 150  # 0–255

    # Custom title bar drag-resize grip thickness
    GRIP_SIZE = 10

    # ── Display backend (V1.36 Phase 4) ─────────────────────────────
    # Default to the pyqtgraph-backed GPU canvas in
    # ``MultiAxisViewer``. Falls back to the legacy QLabel canvas if
    # ``ND2_DISABLE_GPU_DISPLAY=1`` is in the environment, or if
    # ``GpuImageCanvas`` construction raises at runtime.
    USE_GPU_DISPLAY = True
    GPU_DISPLAY_ENV_DISABLE = "ND2_DISABLE_GPU_DISPLAY"

    # ── GPU analysis dispatch (V1.39 Phase 7) ───────────────────────
    # Off by default — the user must opt in via the Performance dialog
    # so analysis outputs stay byte-identical to V1.38 unless they
    # actively choose to route through CuPy / cucim. The env var is
    # the hard override ("never use GPU, even if the dialog is on").
    USE_GPU_ANALYSIS = False
    GPU_ANALYSIS_ENV_DISABLE = "ND2_DISABLE_GPU_ANALYSIS"

    # ── Multi-resolution pyramids (V1.39 Phase 7) ───────────────────
    # On by default — pyramid builds are background jobs that do not
    # block the GUI, and the viewer falls back to level 0 gracefully
    # while the build is in progress. The env var lets the profiling
    # harness pin the viewer to level 0 for like-for-like measurements.
    BUILD_PYRAMIDS = True
    PYRAMID_ENV_DISABLE = "ND2_DISABLE_PYRAMIDS"

    # ── Streaming (overarching viewing setup, V1.66) ────────────────
    # When True, every file opens via the on-demand StreamingDataset (memory-map
    # / lazy reads + bounded cache + prefetch) for uniformly instant, smooth
    # viewing. Eager full-RAM loading is opt-in via FORCED_LOAD_STRATEGY
    # ("eager_full") for workflows that want zero per-frame latency on a small
    # resident file.
    STREAM_ALWAYS = True

    # ── Deep-Z display preview (V1.66) ──────────────────────────────
    # For a streamed deep Z-stack, the viewer projects a bounded number of
    # evenly-spaced Z planes for a faster (approximate) DISPLAY frame; the
    # recipe / export / DVC paths always read the exact full projection.
    # 0 = OFF (exact display). Default 32 keeps deep-stack scrubbing smooth
    # while data stays exact; set 0 for exact display of very deep projections.
    DISPLAY_PREVIEW_Z_PLANES = 32

    # ── 3-D viewer render bounds (V1.66) ────────────────────────────
    # The PyVista volume renderer NEVER hands VTK a full-resolution deep stack
    # (that exhausts GPU/host memory and crashes). It downsamples to a GPU-safe
    # size: XY strided so the larger spatial axis is <= VIEW3D_XY_MAX, Z
    # subsampled to <= VIEW3D_Z_MAX planes, capped at VIEW3D_MAX_VOXELS per
    # channel. Lower these if the 3-D view is slow or unstable on a weak GPU.
    VIEW3D_XY_MAX = 512
    VIEW3D_Z_MAX = 128
    VIEW3D_MAX_VOXELS = 48_000_000

    # ── Resource-aware load strategy (V1.41) ────────────────────────
    # EAGER_MAX_FRACTION: cap on the share of available RAM that the
    # eager-materialization path is allowed to consume. The remaining
    # share is reserved for the OS, the Qt pixmap cache, downstream
    # analysis pipelines, and matplotlib figures.
    # MEMORY_RESERVE_OVERHEAD: safety multiplier applied to the
    # estimated footprint before fit check; soaks up working-set
    # growth during decode (chunked reads, dask threading buffers,
    # PIL/imageio backbuffers).
    # MEMORY_PRESSURE_*: percentage thresholds for the runtime memory
    # monitor. Lower bands trigger soft signals (clear caches);
    # higher bands surface modal warnings and block new ops.
    EAGER_MAX_FRACTION = 0.50
    MEMORY_RESERVE_OVERHEAD = 1.20
    MEMORY_PRESSURE_WARNING_PCT = 80.0
    MEMORY_PRESSURE_CRITICAL_PCT = 90.0
    MEMORY_PRESSURE_EMERGENCY_PCT = 95.0
    MEMORY_PRESSURE_RECOVER_WARNING_PCT = 75.0
    MEMORY_PRESSURE_RECOVER_CRITICAL_PCT = 85.0
    MEMORY_PRESSURE_RECOVER_EMERGENCY_PCT = 90.0
    MEMORY_MONITOR_INTERVAL_MS = 1500

    # Forced LoadStrategy override (empty string = auto pick).
    # When non-empty, must be one of the LoadStrategy enum values:
    # "eager_full", "eager_reduced", or "lazy_cached". Surfaced via
    # the Performance dialog and persisted by ``utils/user_config``.
    FORCED_LOAD_STRATEGY = ""

    # ── Dracula palette (matches CellTracker + Modern_GUI_PyDracula) ──
    BG_PRIMARY = "#282a36"      # main background
    BG_SECONDARY = "#21252b"    # sidebar / top + bottom bars
    BG_TERTIARY = "#2c313c"     # active nav button background
    BG_HOVER = "#343b48"        # hover state
    FG_PRIMARY = "#f8f8f2"      # default text
    FG_SECONDARY = "#b0b0b0"    # subdued text
    ACCENT_PURPLE = "#bd93f9"   # primary accent (CTA, focus, default button)
    ACCENT_PINK = "#ff79c6"     # secondary accent (gradients, highlights)
    ACCENT_GREEN = "#50fa7b"    # success
    ACCENT_CYAN = "#8be9fd"     # info
    ACCENT_ORANGE = "#ffb86c"   # warning
    ACCENT_RED = "#ff5555"      # danger
    ACCENT_YELLOW = "#f1fa8c"
    ACCENT_GOLD = "#ffc83d"     # "previewed" highlight (Pipelines node board)
    ACCENT_WHITE = "#ffffff"    # Checkpoint node (freeze/cache marker)
    BORDER_COLOR = "#44475a"

    # ── Page list ──────────────────────────────────────────────────
    # Tuple format: (key, icon_unicode, label, tooltip)
    PAGES = [
        ("import", "📂", "Import", "Load an ND2 or TIFF file and inspect metadata."),
        ("recipe", "🧪", "Recipe", "Build a processing pipeline (trial / accept / reject)."),
        ("analysis", "🔬", "Analysis", "Run analysis pipelines on loaded data."),
        ("results", "📊", "Results", "Compute measurements from analysis binaries and export CSV / images."),
        ("pipelines", "🧩", "Pipelines", "Build a node graph across processing, analysis, and results."),
        ("batch", "⚡", "Batch", "Run a pipeline template over multiple files and aggregate results."),
        ("export", "💾", "Export", "Export TIFF stacks, RGB composites, movies, and tracked objects."),
    ]

    # Status ladder. Forward navigation can require a minimum status.
    STATUS_ORDER = ["new", "imported", "preprocessed", "ready_to_export"]

    # Per-page minimum status to enter (None = always accessible)
    PAGE_PREREQS = {
        "import": None,
        "recipe": ("imported", "Import a file first (Page 1)."),
        "export": ("imported", "Import a file first (Page 1)."),
        "analysis": ("imported", "Import a file first (Page 1)."),
        "results": ("imported", "Import a file first (Page 1)."),
        "pipelines": ("imported", "Import a file first (Page 1)."),
        "batch": None,
    }
