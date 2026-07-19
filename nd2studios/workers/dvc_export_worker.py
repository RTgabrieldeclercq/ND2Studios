"""Background save / load of a portable DVC bundle (V1.76).

The DVC export/reload writes (or reads) every multipoint's ``DVCResult`` series,
backdrops, masks and per-granule bundles — potentially hundreds of arrays — so the
``numpy.savez_compressed`` / ``numpy.load`` work runs off the GUI thread here (the
"Workers, not async" convention). Gathering the page's DVC stores (export) and
rebuilding them into the viewer + a checkpoint node (import) both stay on the GUI
thread in the page; only the file I/O is threaded.

Both workers use :class:`~nd2studios.workers.base_worker.BaseWorker` signals:
``finished(object)`` carries the output path (export) or the reconstructed bundle
dict (import); ``error(str)`` carries a traceback string.
"""
from __future__ import annotations

from typing import Any, Dict

from nd2studios.workers.base_worker import BaseWorker


class DVCExportWorker(BaseWorker):
    """Write a gathered DVC payload to a ``.nd2dvc`` file off-thread.

    ``payload`` is the keyword dict :func:`nd2studios.backend.dvc_export.save_dvc_bundle`
    expects (``series_by_m``, ``incr_by_m``, ``bg_by_mt``, ``whole_masks_by_m``,
    ``obj_full_by_m``, ``display_mask_by_m``, ``meta``, ``provenance``). The arrays it
    references are the page's own ``DVCResult`` / mask / backdrop arrays — the worker
    only reads them (they are immutable run outputs), so no copy is needed.
    ``finished`` emits the written ``path``.
    """

    def __init__(self, path: str, payload: Dict[str, Any], parent=None) -> None:
        super().__init__(parent)
        self._path = str(path)
        self._payload = payload

    def run_task(self) -> str:
        from nd2studios.backend.dvc_export import save_dvc_bundle
        self.set_status("Exporting DVC results…")
        save_dvc_bundle(self._path, **self._payload)
        return self._path


class DVCImportWorker(BaseWorker):
    """Read a ``.nd2dvc`` file off-thread.

    ``finished`` emits the reconstructed bundle dict from
    :func:`nd2studios.backend.dvc_export.load_dvc_bundle` (the page then repopulates
    its DVC stores + viewer on the GUI thread).
    """

    def __init__(self, path: str, parent=None) -> None:
        super().__init__(parent)
        self._path = str(path)

    def run_task(self) -> Dict[str, Any]:
        from nd2studios.backend.dvc_export import load_dvc_bundle
        self.set_status("Loading DVC results…")
        return load_dvc_bundle(self._path)
