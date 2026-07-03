"""Save / load a :class:`PipelineDoc` (V1.45) — Qt-free.

Mirrors ``backend/recipes.py``: a single JSON file tagged with ``kind`` and a
``schema_version`` so old builds can refuse a newer file gracefully. The full
document — all three stage slices, every node's params / position / enabled
flag, every edge, and the bridge registry — round-trips through
:meth:`PipelineDoc.to_dict` / :meth:`PipelineDoc.from_dict`.
"""
from __future__ import annotations

import json
from typing import Any, Dict

from nd2studios.pipeline_graph.model import GraphSlice, NodeRole, PipelineDoc, Stage
from nd2studios.pipeline_graph.registry_adapter import (
    SPECIAL_REVIEW_OP_KEY, SPECIAL_TRACK_OP_KEY, SPECIAL_VALIDATE_OP_KEY,
    default_params_for, special_specs,
)

# V2 (V1.45 merge): Analysis + Results share one slice; results action nodes now
# live in the analysis slice. V3 (V1.45): the "Validate Tracked Objects" special
# node became the "Track Objects" linker. Older docs are migrated on load.
PIPELINE_VERSION = 3
PIPELINE_KIND = "nd2studios.pipeline"
PIPELINE_EXTENSION = ".nd2s_pipeline.json"


def build_pipeline(doc: PipelineDoc, name: str = "", notes: str = "") -> Dict[str, Any]:
    """Serialize a :class:`PipelineDoc` into a JSON-ready dict."""
    payload = doc.to_dict()
    payload.update(
        {
            "version": PIPELINE_VERSION,
            "schema_version": PIPELINE_VERSION,
            "kind": PIPELINE_KIND,
            "name": name,
            "notes": notes,
        }
    )
    return payload


def save_pipeline(
    path: str, doc: PipelineDoc, name: str = "", notes: str = ""
) -> None:
    """Write ``doc`` to ``path`` as indented JSON."""
    payload = build_pipeline(doc, name=name, notes=notes)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_pipeline(path: str) -> PipelineDoc:
    """Read a ``.nd2s_pipeline.json`` file into a :class:`PipelineDoc`.

    Raises ``ValueError`` with a readable message if the file doesn't look like
    a pipeline produced by this app, or is from a newer schema.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Pipeline file is not a JSON object.")
    if data.get("kind") != PIPELINE_KIND:
        raise ValueError("File is not an ND2Studios pipeline.")
    version = int(data.get("version", data.get("schema_version", 0)))
    if version > PIPELINE_VERSION:
        raise ValueError(
            f"Pipeline version {version} is newer than this build "
            f"supports (max {PIPELINE_VERSION})."
        )
    doc = PipelineDoc.from_dict(data)
    if version < 2:
        doc = _migrate_v1_to_v2(doc)
    if version < 3:
        doc = _migrate_v2_to_v3(doc)
    _normalize_special_categories(doc)
    return doc


def _normalize_special_categories(doc: PipelineDoc) -> None:
    """Re-derive special nodes' category from the current spec, in place.

    The Cell-Tracker Spatial Maps node moved from the SPECIAL category to RESULTS
    (V1.45). Saved graphs store the old category per node, so refresh it from
    :func:`special_specs` on load — purely cosmetic (color / Add-dialog grouping);
    dispatch is keyed on op_key, which is unchanged.
    """
    cat_by_op = {s.op_key: s.effective_category() for s in special_specs()}
    for slice_ in (doc.processing, doc.analysis, doc.results):
        for node in slice_.nodes.values():
            cat = cat_by_op.get(node.op_key)
            if cat is not None:
                node.category = cat


def _migrate_v1_to_v2(doc: PipelineDoc) -> PipelineDoc:
    """Fold a v1 doc's separate Results slice into the merged Analysis slice.

    Results **action** nodes (and edges among them) move into the analysis slice,
    keeping ``stage=RESULTS`` so they still color green. The synthetic Results
    INPUT/OUTPUT bridge nodes are dropped (the merged tab uses the analysis input
    and special sink nodes), along with any edges that touched them.
    """
    res = doc.results
    if not res.nodes:
        return doc
    drop = {nid for nid, n in res.nodes.items()
            if n.role in (NodeRole.INPUT, NodeRole.OUTPUT)}
    for nid, node in res.nodes.items():
        if nid not in drop:
            doc.analysis.nodes[nid] = node
    for eid, edge in res.edges.items():
        if edge.src_node in drop or edge.dst_node in drop:
            continue
        doc.analysis.edges[eid] = edge
    doc.results = GraphSlice(Stage.RESULTS)  # now empty (retained for slice_for)
    doc.schema_version = PIPELINE_VERSION
    return doc


def _migrate_v2_to_v3(doc: PipelineDoc) -> PipelineDoc:
    """Retarget retired ``Validate Tracked Objects`` nodes to ``Track Objects``.

    The interactive validate node was repurposed into the configurable centroid
    linker. Any node still carrying the old ``special:validate_tracks`` op_key is
    retitled and re-keyed to ``special:track_objects`` and seeded with the new
    tracking params (accept/reject now lives in the Review Objects node). The
    Review node is also retitled (``Review Object`` → ``Review Objects``) and
    seeded with its new ``mode`` param.
    """
    track_defaults = default_params_for(SPECIAL_TRACK_OP_KEY)
    review_defaults = default_params_for(SPECIAL_REVIEW_OP_KEY)
    for slice_ in (doc.processing, doc.analysis, doc.results):
        for node in slice_.nodes.values():
            if node.op_key == SPECIAL_VALIDATE_OP_KEY:
                node.op_key = SPECIAL_TRACK_OP_KEY
                node.title = "Track Objects"
                node.params = dict(track_defaults)
            elif node.op_key == SPECIAL_REVIEW_OP_KEY:
                node.title = "Review Objects"
                for k, v in review_defaults.items():
                    node.params.setdefault(k, v)
    doc.schema_version = PIPELINE_VERSION
    return doc
