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

from nd2studios.pipeline_graph.model import (
    Edge, GraphSlice, NodeRole, PipelineDoc, Stage, new_id,
)
from nd2studios.pipeline_graph.registry_adapter import (
    SPECIAL_REVIEW_OP_KEY, SPECIAL_TRACK_OP_KEY, SPECIAL_VALIDATE_OP_KEY,
    default_params_for, special_specs,
)

# V2 (V1.45 merge): Analysis + Results share one slice; results action nodes now
# live in the analysis slice. V3 (V1.45): the "Validate Tracked Objects" special
# node became the "Track Objects" linker. V4 (V1.49): edges gained a ``kind``
# ("structural"/"loop") + ``params`` for loop connectors; older edges default to
# structural on load (no data migration needed). V5/V6 (V1.61 overhaul): Processing
# is folded into the merged slice — the two sub-tabs (Processing + Analysis) become
# one scene. V6 also drops the intermediate Processing-OUTPUT / Analysis-INPUT
# bridge nodes: the graph is one connected chain from a single universal input
# through processing into analysis, ending in output nodes. Older docs are
# migrated on load (bridge halves reconnected).
PIPELINE_VERSION = 6
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
    if version < 6:
        doc = _migrate_to_connected(doc)
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


def _migrate_to_connected(doc: PipelineDoc) -> PipelineDoc:
    """Merge the Processing + Analysis sub-tabs into ONE connected graph (V1.61).

    Two steps:

    1. **Fold** every Processing node + edge into the merged (analysis) slice, so
       all nodes live in one :class:`GraphSlice`. Node ``stage`` attributes are
       preserved (they still drive execution dispatch: enhancement recipe vs.
       analysis pipeline).
    2. **Drop the bridge nodes and reconnect.** The old model had a Processing
       ``OUTPUT`` node and an Analysis ``INPUT`` node, two *disconnected*
       components joined only through ``record.recipe``. The new model is a single
       chain: one universal input → processing → analysis → output. So the tail of
       the processing chain (the node that fed the Processing OUTPUT, else the
       universal file input) is wired directly to whatever the Analysis INPUT node
       fed, and both bridge nodes are removed.

    Lossless for the common case (a file input + a processing chain + an analysis
    chain). If the saved graph has no universal input, the Analysis INPUT node is
    kept (promoted implicitly by staying) so analysis nodes never end up orphaned.
    """
    # 1. Fold Processing into the merged slice.
    proc = doc.processing
    if proc.nodes or proc.edges:
        for nid, node in proc.nodes.items():
            doc.analysis.nodes.setdefault(nid, node)
        for eid, edge in proc.edges.items():
            doc.analysis.edges.setdefault(eid, edge)
        doc.processing = GraphSlice(Stage.PROCESSING)

    sl = doc.analysis

    def _is_proc(n):
        return n.stage is Stage.PROCESSING

    universal_in = next(
        (n for n in sl.nodes.values()
         if n.role is NodeRole.INPUT and _is_proc(n)), None)
    proc_outs = [n for n in sl.nodes.values()
                 if n.role is NodeRole.OUTPUT and _is_proc(n)]
    analysis_ins = [n for n in sl.nodes.values()
                    if n.role is NodeRole.INPUT and n.stage is not Stage.PROCESSING]

    # 2. Resolve the processing exit (tail enhancement feeding a Processing OUTPUT,
    #    else the universal input) as the new source for the analysis heads.
    exit_node_id = None
    exit_port_id = None
    for po in proc_outs:
        ip = po.inputs[0] if po.inputs else None
        if ip is None:
            continue
        fed = next((e for e in sl.edges.values()
                    if e.dst_node == po.id and e.dst_port == ip.id
                    and e.kind != "loop"), None)
        if fed is not None:
            exit_node_id, exit_port_id = fed.src_node, fed.src_port
            break
    if exit_node_id is None and universal_in is not None and universal_in.outputs:
        exit_node_id = universal_in.id
        exit_port_id = universal_in.outputs[0].id

    # Reconnect the analysis heads to the processing exit, then drop the Analysis
    # INPUT bridge nodes. Skip reconnection (and keep the input) if we have no exit
    # source, so analysis is never left dangling.
    if exit_node_id is not None:
        for ai in analysis_ins:
            heads = [(e.dst_node, e.dst_port) for e in list(sl.edges.values())
                     if e.src_node == ai.id and e.kind != "loop"]
            for dn, dp in heads:
                sl.add_edge(Edge(new_id("e"), exit_node_id, exit_port_id, dn, dp))
            sl.remove_node(ai.id)

    # Drop the Processing OUTPUT bridge nodes (their recipe role is now served by
    # the tail enhancement node via _graph_recipe).
    for po in proc_outs:
        sl.remove_node(po.id)

    doc.schema_version = PIPELINE_VERSION
    return doc
