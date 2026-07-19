"""Op-key constants for the V1.70 Granule Separation nodes.

These five **special** nodes separate a 3-D point cloud of bead centroids into the
hydrogel *granules* they belong to, then define, mask, view, and rim each granule.
They chain:

    [Bead Detect] ─DATA─▶ [Cluster] ─DATA─▶ [Tessellate] ─ANY─▶ [Mask] ─┬─ANY─▶ [Boundary]
                                                                        └──── "View in 3-D"

Kept in their own module (imported by both ``registry_adapter.py`` and
``pages/pipelines_page.py``) so the two shared files don't each hard-code the op-key
strings — mirroring how the DVC / Mask3D op keys are consumed in two places.

The nodes reuse the shipped V1.65–V1.68 back half: the mask node publishes per-granule
``(Z, H, W) bool`` volumes shaped exactly like ``record._mask3d_by_m[m][t]``, so
``object_scope.iter_objects``, the Frame/Object scope lever, DVC-on-object, and the
PyVista surface stack consume them unchanged.
"""
from __future__ import annotations

# Bead Detection (V1.70 P1) — detect bead centroids in the wired channel's raw
# (Z,H,W) volume; the app's first ``centroid_z_px`` writer. IMAGE in (rainbow
# channel), DATA out (point-cloud rows).
SPECIAL_BEAD_DETECT_OP_KEY = "special:bead_detect"
# Granule Clustering (V1.70 P2) — GMM (full covariance) with a BIC sweep over the
# user-relaxed granule count. DATA in/out (rows gain a ``granule_id`` column).
SPECIAL_GRANULE_CLUSTER_OP_KEY = "special:granule_cluster"
# Granule Tessellation + density-merge (V1.70 P3) — per-granule alpha-shape OR global
# Voronoi boundaries + a region-adjacency density-ratio merge producing FINAL labels.
SPECIAL_GRANULE_TESSELLATE_OP_KEY = "special:granule_tessellate"
# Granule Volume Mask (V1.70 P4) — voxelize the tessellated boundaries onto the
# confocal grid with SDF-Gaussian smoothing. The object-producing node.
SPECIAL_GRANULE_MASK_OP_KEY = "special:granule_mask"
# Granule Boundary Extraction (V1.70 P5) — outward band of N voxels into anything
# non-self (background + neighbour granules), by dilation or EDT.
SPECIAL_GRANULE_BOUNDARY_OP_KEY = "special:granule_boundary"

# Convenience grouping (palette / dispatch iteration).
GRANULE_OP_KEYS = (
    SPECIAL_BEAD_DETECT_OP_KEY,
    SPECIAL_GRANULE_CLUSTER_OP_KEY,
    SPECIAL_GRANULE_TESSELLATE_OP_KEY,
    SPECIAL_GRANULE_MASK_OP_KEY,
    SPECIAL_GRANULE_BOUNDARY_OP_KEY,
)
