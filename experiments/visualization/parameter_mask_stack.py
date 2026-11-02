"""Isometric stack of randomly sampled parameter masks laid flat, all model families in one stack.

Masks are drawn with experiment 1's training geometry and mask-randomizer settings
and collated by ``ModelClass.collate``, so every sheet sits on the model class's
global grid (rows: intercept, main-effect blocks, interaction blocks; columns: the
union of all families' intrinsic parameters). Active cells take their family's hue,
shaded by term kind; inactive cells, including parameters outside the family, are
gray, and rows of terms absent from a draw's design are a lighter gray.
``--prior abundant`` (default) samples denser designs from the same generator;
``--prior training`` reproduces the training distribution.

Usage:
    python -m experiments.visualization.parameter_mask_stack --num_per_family 6 --seed 1
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import PolyCollection

from cogformer.simulator import ModelFamily, ModelClass
from cogformer.simulator.benchmarks.ssms.ddm import DDM
from cogformer.simulator.benchmarks.ssms.rdm import RDM
from cogformer.simulator.benchmarks.ssms.cdm import CDM
from cogformer.simulator.benchmarks.ssms.priors import ddm_priors, rdm_priors, cdm_priors
from cogformer.simulator.benchmarks.ssms.link_funs import ddm_link_fun, rdm_link_fun, cdm_link_fun
from cogformer.utils.plot_utils import bf_colors, cogformer_cm_colors, cogformer_fm_colors

from ..experiment_1.model_class_cf_train import _MODELS_WITH_PI, GLOBAL_PARAMS


FIG_DIR = Path("./outputs/figures/visualization")

FAMILIES = {
    "DDM": (DDM, ddm_priors, ddm_link_fun, ["v", "a", "z", "tau"]),
    "RDM": (RDM, rdm_priors, rdm_link_fun, ["v", "v_diff", "a", "tau"]),
    "CDM": (CDM, cdm_priors, cdm_link_fun, ["v", "v_theta", "a", "tau"]),
}

# Design-config knobs. "training" is experiment 1's sampler (free_prob and
# intercept_prob None -> Beta(2, 2) per design); "abundant" keeps the same generator
# but pushes every knob toward full designs so the term zones read clearly. A None
# count means "the maximum" (--max_num_regressors / --max_num_categories).
PRESETS = {
    "training": {"min_num_regressors": 0, "free_prob": None, "intercept_prob": None,
                 "discrete_prob": 0.5, "min_num_categories": 2},
    "abundant": {"min_num_regressors": None, "free_prob": 0.8, "intercept_prob": 0.9,
                 "discrete_prob": 0.8, "min_num_categories": None},
}

# One hue per family, shaded dark -> light for intercept -> main effect -> interaction.
# bf_colors' intercept and main effect are both near-black green, so the RDM main
# effect is lifted to a mid green to keep the two zones apart.
FAMILY_COLORS = {
    "DDM": cogformer_cm_colors(),
    "RDM": {**bf_colors(), "main_effect": "#4F8A3C"},
    "CDM": cogformer_fm_colors(),
}

INACTIVE_GRAY = "#c8c8c8"
PADDING_GRAY = "#efefef"
COS30, SIN30 = np.cos(np.pi / 6), np.sin(np.pi / 6)


def build_model_class() -> ModelClass:
    """ModelClass with experiment 1's families and mask-randomizer settings."""
    families = {}
    link_funs = {}
    for name, (model_cls, prior_fn, link_fn, free) in FAMILIES.items():
        cfg = _MODELS_WITH_PI[name.lower()]
        families[name] = ModelFamily(
            model=model_cls(), name=name, prior=prior_fn(),
            intrinsic_params=cfg["intrinsic_params"],
            mask_randomizer_kwargs={
                "free_intrinsics": free,
                "fixed_intrinsics": ["s_v", "s_tau"],
                "fixed_values": cfg["default_fixed_values"],
                "global_params": GLOBAL_PARAMS,
            },
            num_workers=1,
        )
        link_funs[name] = link_fn()
    return ModelClass(model_families=families, link_funs=link_funs,
                      train_contexts={"RDM": RDM.build_random_context}, num_workers=1)


def sample_family(mc: ModelClass, name: str, num_masks: int, args, preset: dict) -> dict:
    """Draw ``num_masks`` independent designs from one family on the global grid.

    Mirrors ``ModelClass.batch_sample``'s per-element draw, but also passes the
    design-config knobs it leaves at their defaults (free_prob, intercept_prob,
    discrete_prob, min_num_categories), so a preset can make designs denser while
    the configs still come from build_random_design_config.
    """
    mf = mc.model_families[name]
    ctx_fn = mc.train_contexts.get(name)
    mask_kwargs = {**mf.mask_randomizer_kwargs, "intercept_prob": preset["intercept_prob"]}
    num_obs = np.random.randint(args.min_num_obs, args.max_num_obs + 1)
    min_regressors = preset["min_num_regressors"]
    min_regressors = args.max_num_regressors if min_regressors is None else min_regressors
    min_categories = preset["min_num_categories"]
    min_categories = args.max_num_categories if min_categories is None else min_categories
    items = []
    for _ in range(num_masks):
        num_regressors = np.random.randint(min_regressors, args.max_num_regressors + 1)
        # mf.sample builds the model's default context itself when given None
        context = ctx_fn(num_obs) if ctx_fn is not None else None
        item = mf.sample(
            num_obs=num_obs,
            num_regressors=num_regressors,
            max_num_regressors=args.max_num_regressors,
            max_num_categories=args.max_num_categories,
            min_num_categories=min_categories,
            context=context,
            mask_randomizer_kwargs=mask_kwargs,
            discrete_prob=preset["discrete_prob"],
            free_prob=preset["free_prob"],
            keep_intercept=True,
            flatten_param_outputs=False,
            add_interaction=True,
            link_fun=mc.link_funs[name],
        )
        item.update(model_id=mc.model_index[name], num_obs=num_obs, num_regressors=num_regressors,
                    max_num_regressors=args.max_num_regressors,
                    max_num_categories=args.max_num_categories)
        items.append(item)
    batch = mc.collate(items, flatten_param_outputs=False)
    return {k: batch[k] for k in ("param_masks", "param_matrices", "design_matrices",
                                  "regressor_masks", "design_configs", "num_obs")}


def row_kinds(max_num_regressors: int, max_num_categories: int) -> np.ndarray:
    """Term kind per grid row, in build_parameter_mask order."""
    bw = max_num_categories - 1
    n_inter = max_num_regressors * (max_num_regressors - 1) // 2
    return np.array(["intercept"] + ["main"] * (max_num_regressors * bw)
                    + ["interaction"] * (n_inter * bw ** 2))


def project(x, y, z):
    """Isometric projection of a sheet lying flat: x = rows (down-right),
    y = columns (down-left), z = height of the sheet in the stack."""
    return COS30 * (x - y), z - SIN30 * (x + y)


def column_y(j, num_cols):
    """Plane coordinate of column ``j``'s near edge in the base (unturned) layout:
    columns run back to front, so the first column (v) lies along the front edge."""
    return num_cols - j


def layer_height(k, num_layers, layer_gap):
    """Height of layer ``k``; layer 0 (the first mask) is on top."""
    return (num_layers - 1 - k) * layer_gap


def rotate_plane(x, y, quarter_turns, num_rows, num_cols):
    """Base-layout plane point (x along rows, y along columns) turned by
    ``quarter_turns`` x 90 degrees about the vertical, re-anchored so the sheet's
    footprint stays at the origin. One turn maps (x, y) -> (y, num_rows - x): the
    intercept row comes to the front-left edge and column v to the front-right."""
    for _ in range(quarter_turns % 4):
        x, y = y, num_rows - x
        num_rows, num_cols = num_cols, num_rows
    return x, y


def draw_stack(ax, faces, layer_gap, cuts=(), quarter_turns=0):
    """Draw one stack of flat sheets, bottom to top, with the first sheet on top.
    ``faces`` is (layers, rows, cols, 4) RGBA per cell; each cell's white grid line
    takes its face's alpha, so translucent cells stay see-through. The layout is
    turned by ``quarter_turns`` x 90 degrees in the horizontal plane (see
    rotate_plane).

    ``cuts`` are row indices where a vertical section plane slices the stack. Nothing
    on the far side of a vertical plane can occlude anything on its near side in
    this view, so drawing the stack slab by slab from back to front (each bottom to
    top) with each plane's back and bottom edges in between gives correct
    occlusion. The top and front edges can never be occluded, so they go on last."""
    num_layers, num_rows, num_cols = faces.shape[:3]
    bounds = [0, *cuts, num_rows]
    z_top = layer_height(0, num_layers, layer_gap)

    def rot(x, y):
        return rotate_plane(x, y, quarter_turns, num_rows, num_cols)

    def to_screen(x, y, z):
        return project(*rot(x, y), z)

    # Slabs back to front: depth along the view direction is x + y in the turned plane.
    slabs = sorted(zip(bounds[:-1], bounds[1:]),
                   key=lambda s: sum(rot((s[0] + s[1]) / 2, num_cols / 2)))
    for n, (r0, r1) in enumerate(slabs):
        if n > 0:
            (shared,) = set(slabs[n - 1]) & {r0, r1}
            draw_section_edges(ax, shared, num_cols, z_top, ("back", "bottom"), rot)
        for k in reversed(range(num_layers)):
            z = layer_height(k, num_layers, layer_gap)
            polys, cell_faces, edges = [], [], []
            for i in range(r0, r1):
                for j in range(num_cols):
                    y = column_y(j, num_cols)
                    corners = [(i, y - 1), (i + 1, y - 1), (i + 1, y), (i, y)]
                    polys.append([to_screen(x, y, z) for x, y in corners])
                    cell_faces.append(faces[k, i, j])
                    edges.append((1.0, 1.0, 1.0, faces[k, i, j, 3]))
            ax.add_collection(PolyCollection(polys, facecolors=cell_faces, edgecolors=edges,
                                             linewidths=0.5))

    for x in cuts:
        draw_section_edges(ax, x, num_cols, z_top, ("top", "front"), rot)


def draw_section_edges(ax, x, num_cols, height, which, rot, margin=1.2):
    """Dash-dot outline edges (``which``: back, bottom, top, front) of the vertical
    section plane at row ``x``, placed through ``rot`` (base layout -> turned plane).
    The plane spans exactly the bottom to the top sheet and overshoots only at its
    two horizontal ends, so its top and bottom edges lie on the boundary between
    rows x-1 and x of the top and bottom sheets; overshooting in height would read,
    in this projection, as a cut at a different row. Outline only: a fill would
    wash out every cell behind the plane."""
    back, front = sorted([rot(x, -margin), rot(x, num_cols + margin)], key=sum)
    back_bottom, front_bottom = (*back, 0), (*front, 0)
    back_top, front_top = (*back, height), (*front, height)
    edges = {"back": (back_bottom, back_top), "bottom": (back_bottom, front_bottom),
             "top": (back_top, front_top), "front": (front_bottom, front_top)}
    for name in which:
        (x0, y0), (x1, y1) = (project(*p) for p in edges[name])
        ax.plot([x0, x1], [y0, y1], color="#333333", linewidth=0.7, linestyle="-.")


def term_zones(kinds):
    """(kind, first row, end row) for the intercept, main-effect and interaction zones."""
    zones = []
    for kind in ("intercept", "main", "interaction"):
        rows = np.where(kinds == kind)[0]
        zones.append((kind, rows[0], rows[-1] + 1))
    return zones


def structure_faces(masks, exists, inactive_alpha, padding_alpha):
    """RGBA faces for the cells every stack figure shares: translucent gray for
    inactive cells (term in the design, coefficient off, including parameters the
    family does not have) and a lighter gray for padding (term absent from the
    design). Active cells are left fully transparent for the caller to fill;
    returns (faces, active)."""
    faces = np.zeros(masks.shape + (4,))
    faces[exists & (masks == 0)] = mcolors.to_rgba(INACTIVE_GRAY, inactive_alpha)
    faces[~exists] = mcolors.to_rgba(PADDING_GRAY, padding_alpha)
    return faces, exists & (masks > 0)


def mask_faces(stack, kinds, inactive_alpha, padding_alpha):
    """Mask figure faces: active cells in their family's shade for the term kind."""
    faces, active = structure_faces(stack["masks"], stack["exists"], inactive_alpha, padding_alpha)
    kind_key = {"intercept": "intercept", "main": "main_effect", "interaction": "interaction"}
    for k, i, j in zip(*np.nonzero(active)):
        faces[k, i, j] = mcolors.to_rgba(FAMILY_COLORS[stack["families"][k]][kind_key[kinds[i]]])
    return faces


def stack_from_draws(draws):
    """Assemble the drawn stack from saved draws (the dict written to, or an npz read
    from, ``*_draws.npz``): families concatenated in FAMILIES order, put in the
    saved stack order, restricted to the shown parameter columns."""
    order, shown = draws["stack_order"], draws["shown_params"].astype(bool)
    families = np.concatenate([[name] * len(draws[f"{name}_param_masks"]) for name in FAMILIES])

    def stacked(key):
        return np.concatenate([draws[f"{name}_{key}"] for name in FAMILIES])[order]

    masks = stacked("param_masks")[:, :, shown]
    # A cell exists when its term is in the draw's design, whatever the column.
    exists = np.broadcast_to(stacked("regressor_masks").astype(bool)[:, :, None], masks.shape)
    return {"masks": masks, "exists": exists,
            "values": stacked("param_matrices")[:, :, shown], "families": families[order]}


def render_stack(faces, kinds, layer_gap, rotation, figsize=(7, 7)):
    """Figure with one stack of ``faces``, section cuts at the term-zone boundaries."""
    fig, ax = plt.subplots(figsize=figsize)
    cuts = [x0 for _, x0, _ in term_zones(kinds)[1:]]
    draw_stack(ax, faces, layer_gap, cuts=cuts, quarter_turns=rotation // 90)
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.axis("off")
    return fig, ax


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_per_family", type=int, default=6)
    parser.add_argument("--rotation", type=int, choices=[0, 90, 180, 270], default=270,
                        help="Turn of the stack in the horizontal plane (see rotate_plane). 90 "
                             "puts the intercept row on the front-left edge and column v on the "
                             "front-right; 270 turns the other way.")
    parser.add_argument("--order", choices=["shuffled", "grouped"], default="grouped",
                        help="Interleave families through the stack (as a model-class batch "
                             "does), or keep each family's sheets together.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max_num_regressors", type=int, default=2)
    parser.add_argument("--max_num_categories", type=int, default=3)
    parser.add_argument("--min_num_obs", type=int, default=200)
    parser.add_argument("--max_num_obs", type=int, default=1000)
    parser.add_argument("--stack_height", type=float, default=7.0,
                        help="Height from bottom to top sheet; the gap is shared out over all sheets.")
    parser.add_argument("--inactive_alpha", type=float, default=0.15)
    parser.add_argument("--padding_alpha", type=float, default=0.06)
    parser.add_argument("--prior", choices=list(PRESETS), default="abundant",
                        help="Design-config knobs: experiment 1's training sampler, or a denser one.")
    parser.add_argument("--out", type=str, default="parameter_mask_stack")
    args = parser.parse_args()
    num_layers = args.num_per_family * len(FAMILIES)
    layer_gap = args.stack_height / max(num_layers - 1, 1)
    preset = PRESETS[args.prior]

    np.random.seed(args.seed)
    mc = build_model_class()

    kinds = row_kinds(args.max_num_regressors, args.max_num_categories)

    saved = {}
    for name in FAMILIES:
        draws = sample_family(mc, name, args.num_per_family, args, preset)

        for key in ("param_masks", "param_matrices", "design_matrices", "regressor_masks", "num_obs"):
            saved[f"{name}_{key}"] = draws[key]

    order = (np.random.permutation(num_layers) if args.order == "shuffled"
             else np.arange(num_layers))
    # Hide the global contamination column (pi) from the drawing only: it is still
    # sampled as in training and kept in the saved draws.
    shown = ~np.isin(mc.global_param_names, GLOBAL_PARAMS)
    # Layout goes with the draws so companion figures (coefficient_matrix_stack)
    # redraw exactly this stack.
    saved.update(global_param_names=mc.global_param_names, row_kinds=kinds, stack_order=order,
                 shown_params=shown, layer_gap=layer_gap, rotation=args.rotation,
                 inactive_alpha=args.inactive_alpha, padding_alpha=args.padding_alpha)
    stack = stack_from_draws(saved)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, _ = render_stack(mask_faces(stack, kinds, args.inactive_alpha, args.padding_alpha),
                          kinds, layer_gap, args.rotation)
    fig.tight_layout()

    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{args.out}.{ext}", dpi=300, bbox_inches="tight")
    np.savez(FIG_DIR / f"{args.out}_draws.npz", stack_families=stack["families"], **saved)
    print(f"saved {FIG_DIR / args.out}.{{pdf,png}} and _draws.npz")


if __name__ == "__main__":
    main()
