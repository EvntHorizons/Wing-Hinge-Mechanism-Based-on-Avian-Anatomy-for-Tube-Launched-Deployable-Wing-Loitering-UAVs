"""
hinge_visualize.py
==================
Rich PyVista 3D re-visualisation of the lug co-optimised hinge assembly.

Features
--------
    • Stepped pin surface-coloured by von Mises stress (mission worst-case loads)
    • Lug plates at optimised B_lug — ghost wireframe shows original width
    • Hole cylinders cut through each plate for physical accuracy
    • Load arrows: lift (firebrick), reactions (steelblue), hard-stop impact (crimson)
    • Labelled σ_VM colorbar and dimension/mass annotations
    • Dual-viewport: isometric (left)  +  orthographic side profile (right)

Outputs:  lug_hinge_3d.png
"""

import json
import pathlib
import sys
import numpy as np
import pyvista as pv

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from discreteload_hinge_sim import (
    SteppedPin,
    T_PLATE, H_GAP, D_HOLE, L_PIN,
    Z_TOP_PLATE, Z_TOP_WING, Z_BOT_WING, Z_BOT_PLATE,
    Z_STEP_1, Z_STEP_2,
    M_combined, sigma_bend, tau_shear, vm,
    B_LUG_EFF, F_STOP, V_DESIGN,
)

L_LUG = 2.5 * D_HOLE       # lug fitting chord-wise depth (matches lug_coopt.py)

# ==========================================================================
# DATA LOADING
# ==========================================================================
def _load_json(name):
    p = pathlib.Path(__file__).parent / name
    with open(p) as f:
        return json.load(f)


def load_design():
    d   = _load_json("lug_optimized_pin.json")
    pin = SteppedPin(d["material"], d["d_end_m"], d["d_mid_m"])
    return pin, float(d["B_lug_m"]), d


def load_mission_loads():
    try:
        d = _load_json("mission_optimized_pin.json")
        return float(d["V_worst_N"]), float(d["M_chord_worst_Nm"])
    except FileNotFoundError:
        from discreteload_hinge_sim import V_DESIGN as VD, M_DRAG as MD
        return VD, MD


# ==========================================================================
# STRESS-COLOURED PIN MESH
# ==========================================================================
def build_pin_mesh(pin, V, M_chord, n_segs=120, n_sides=56):
    """
    Assemble the stepped pin as n_segs short cylinders, each carrying
    the average σ_VM at its mid-plane as a point scalar.
    Returns (merged PolyData, global_min_MPa, global_max_MPa).
    """
    z_bounds = np.linspace(0, L_PIN, n_segs + 1)
    meshes   = []
    vm_means = []

    for i in range(n_segs):
        z0, z1 = z_bounds[i], z_bounds[i + 1]
        zc     = 0.5 * (z0 + z1)
        d      = pin.diameter_at(zc)
        M      = M_combined(zc, V, M_chord)
        sb     = sigma_bend(M, d)
        tv     = tau_shear(V, d)
        vm_val = vm(sb, tv) / 1e6        # MPa

        cyl = pv.Cylinder(
            center    = (0.0, 0.0, zc),
            direction = (0.0, 0.0, 1.0),
            radius    = d / 2.0,
            height    = z1 - z0 + 1e-9,  # tiny overlap prevents dark seams
            resolution= n_sides,
            capping   = True,
        )
        cyl["sigma_VM_MPa"] = np.full(cyl.n_points, vm_val)
        meshes.append(cyl)
        vm_means.append(vm_val)

    combined = pv.MultiBlock(meshes).combine()
    arr      = np.array(vm_means)
    return combined, float(arr.min()), float(arr.max())


# ==========================================================================
# SCENE ASSEMBLY
# ==========================================================================
PLATE_DEFS = [
    (0,                    T_PLATE,              "navy",   "PLATE (top)"),
    (T_PLATE,              2 * T_PLATE,           "#6B22A0","Wing (top)"),
    (2 * T_PLATE + H_GAP,  3 * T_PLATE + H_GAP,  "#6B22A0","Wing (bot)"),
    (3 * T_PLATE + H_GAP,  L_PIN,                "navy",   "PLATE (bot)"),
]


def add_plates(pl, B_lug):
    """Add lug + wing plates with hole cylinders. B_lug = optimised width."""
    for z0, z1, color, _ in PLATE_DEFS:
        is_lug = color == "navy"
        b      = B_lug if is_lug else B_lug  # both use same footprint for symmetry

        # Solid plate
        box = pv.Box(bounds=(-b / 2, b / 2, -L_LUG / 2, L_LUG / 2, z0, z1))
        pl.add_mesh(box, color=color, opacity=0.28,
                    show_edges=True, edge_color="#111111", line_width=0.8)

        # White hole cylinder (visual hole cut through plate)
        hole = pv.Cylinder(
            center    = (0.0, 0.0, (z0 + z1) / 2),
            direction = (0.0, 0.0, 1.0),
            radius    = D_HOLE / 2,
            height    = (z1 - z0) * 1.02,
            resolution= 48,
        )
        pl.add_mesh(hole, color="white", opacity=1.0, smooth_shading=True)


def add_ghost_lug(pl):
    """Wireframe outline of original over-wide lug for before/after comparison."""
    for z0, z1, _, _ in PLATE_DEFS:
        ghost = pv.Box(bounds=(
            -B_LUG_EFF / 2, B_LUG_EFF / 2,
            -L_LUG / 2,      L_LUG / 2,
            z0,               z1,
        ))
        pl.add_mesh(ghost, style="wireframe", color="#888888",
                    line_width=1.2, opacity=0.55)


def add_load_arrows(pl, V, B_lug):
    """Lift arrows on wing plates, reaction arrows on lug plates, impact arrow."""
    reach = B_lug * 0.55      # arrow tail offset from pin centre
    scale = B_lug * 0.45      # arrow length

    # Lift (shear) — wing plates, +Y direction
    for z_load in (Z_TOP_WING, Z_BOT_WING):
        arr = pv.Arrow(
            start     = (0.0, -(reach + scale), z_load),
            direction = (0.0, 1.0, 0.0),
            scale     = scale,
            tip_length= 0.28, tip_radius=0.10, shaft_radius=0.035,
        )
        pl.add_mesh(arr, color="#C0392B", smooth_shading=True)

    # Reactions — lug plates, −Y direction
    for z_rx in (Z_TOP_PLATE, Z_BOT_PLATE):
        arr = pv.Arrow(
            start     = (0.0, reach + scale, z_rx),
            direction = (0.0, -1.0, 0.0),
            scale     = scale,
            tip_length= 0.28, tip_radius=0.10, shaft_radius=0.035,
        )
        pl.add_mesh(arr, color="#2980B9", smooth_shading=True)

    # Hard-stop impact — axial (+Z), scaled by F_STOP / V_DESIGN ratio
    impact_scale = scale * min(F_STOP / V_DESIGN, 6.0) * 0.18
    arr_impact = pv.Arrow(
        start     = (0.0, 0.0, -impact_scale * 1.1),
        direction = (0.0, 0.0, 1.0),
        scale     = impact_scale,
        tip_length= 0.22, tip_radius=0.12, shaft_radius=0.05,
    )
    pl.add_mesh(arr_impact, color="#8B0000", smooth_shading=True)


# ==========================================================================
# ANNOTATION TEXT
# ==========================================================================
def annotation_lines(pin, B_lug, data, V_w, vm_max):
    saving_pct = 100 * data["lug_mass_saving_g"] / data["mass_total_original_g"]
    return (
        f"Material : {pin.matkey}\n"
        f"d_end    : {pin.d_end * 1000:.2f} mm\n"
        f"d_mid    : {pin.d_mid * 1000:.2f} mm\n"
        f"B_lug    : {B_lug * 1000:.2f} mm  (was {B_LUG_EFF * 1000:.1f} mm)\n"
        f"Pin mass : {data['mass_pin_g']:.3f} g\n"
        f"Lug mass : {data['mass_lug_g']:.3f} g  "
        f"(-{data['lug_mass_saving_g']:.3f} g, -{saving_pct:.0f}%)\n"
        f"Total    : {data['mass_total_g']:.3f} g\n"
        f"Peak σVM : {vm_max:.1f} MPa  @  V={V_w:.1f} N\n"
        f"F_stop   : {F_STOP:.0f} N  ({F_STOP/V_DESIGN:.1f}× aero)"
    )


# ==========================================================================
# CAMERA HELPERS
# ==========================================================================
def _iso_camera(scene_centre, scene_radius, el_deg=28, az_deg=45):
    """Return a camera_position list for an isometric-ish perspective view."""
    el  = np.radians(el_deg)
    az  = np.radians(az_deg)
    r   = scene_radius * 3.5
    cx, cy, cz = scene_centre
    cam = (
        cx + r * np.cos(el) * np.sin(az),
        cy + r * np.cos(el) * np.cos(az),
        cz + r * np.sin(el),
    )
    return [cam, scene_centre, (0.0, 0.0, 1.0)]


def _side_camera(scene_centre, scene_radius):
    """Orthographic side view: camera on +X axis, looking toward origin."""
    cx, cy, cz = scene_centre
    r = scene_radius * 3.5
    return [(cx + r, cy, cz), scene_centre, (0.0, 0.0, 1.0)]


# ==========================================================================
# MAIN
# ==========================================================================
def main(save="lug_hinge_3d.png"):
    pin, B_lug, data = load_design()
    V_w, M_w         = load_mission_loads()

    print(f"Rendering: {pin.matkey}  B_lug={B_lug*1000:.2f}mm  "
          f"V={V_w:.1f}N  M={M_w:.4f}N·m")

    pin_mesh, vm_min, vm_max = build_pin_mesh(pin, V_w, M_w)
    print(f"  σ_VM range: {vm_min:.2f} – {vm_max:.2f} MPa")

    # Scene geometry for camera placement
    scene_cx, scene_cy = 0.0, 0.0
    scene_cz = L_PIN / 2
    scene_r  = max(L_PIN, B_lug) * 1.2

    # ------------------------------------------------------------------
    # Dual-viewport plotter
    # ------------------------------------------------------------------
    pl = pv.Plotter(
        shape       = (1, 2),
        window_size = (2400, 1100),
        off_screen  = True,
    )
    pl.set_background("white")

    scalar_bar_args = dict(
        title          = "σ_VM (MPa)",
        position_x     = 0.03,
        position_y     = 0.06,
        width          = 0.28,
        height         = 0.06,
        color          = "black",
        title_font_size= 16,
        label_font_size= 13,
        n_labels       = 5,
        fmt            = "%.1f",
    )

    # ==============================================================
    # LEFT VIEWPORT — isometric perspective
    # ==============================================================
    pl.subplot(0, 0)

    add_ghost_lug(pl)
    add_plates(pl, B_lug)
    add_load_arrows(pl, V_w, B_lug)

    pl.add_mesh(
        pin_mesh,
        scalars       = "sigma_VM_MPa",
        cmap          = "plasma",
        smooth_shading= True,
        clim          = [vm_min, vm_max],
        show_scalar_bar= True,
        scalar_bar_args= scalar_bar_args,
    )

    # Plate boundary lines along pin axis (subtle wireframe ring at step joints)
    for z_bnd in (Z_TOP_PLATE, Z_TOP_WING, Z_BOT_WING, Z_BOT_PLATE,
                  Z_STEP_1, Z_STEP_2):
        d_ring = pin.diameter_at(z_bnd) * 1.01
        ring   = pv.Circle(radius=d_ring / 2, resolution=64)
        ring.translate([0.0, 0.0, z_bnd], inplace=True)
        pl.add_mesh(ring, style="wireframe", color="#333333",
                    line_width=1.0, opacity=0.6)

    # Ghost-lug label arrow (show the B_LUG_EFF width)
    lug_y_ghost = B_LUG_EFF / 2 * 1.05
    pl.add_text(
        annotation_lines(pin, B_lug, data, V_w, vm_max),
        position   = "lower_right",
        font_size  = 11,
        color      = "black",
        font       = "courier",
    )
    pl.add_text(
        "Grey wireframe = original lug  (B = 12.70 mm)\n"
        "Solid plates   = optimised lug (B =  9.53 mm)\n"
        "Crimson arrow  = hard-stop impact (F_stop)",
        position  = "lower_left",
        font_size = 10,
        color     = "dimgrey",
    )

    pl.camera_position = _iso_camera(
        (scene_cx, scene_cy, scene_cz), scene_r, el_deg=28, az_deg=40
    )
    pl.enable_anti_aliasing("msaa")

    # ==============================================================
    # RIGHT VIEWPORT — orthographic side profile (looking from +X)
    # ==============================================================
    pl.subplot(0, 1)
    pl.enable_parallel_projection()   # orthographic for engineering side view

    add_plates(pl, B_lug)

    pl.add_mesh(
        pin_mesh,
        scalars        = "sigma_VM_MPa",
        cmap           = "plasma",
        smooth_shading = True,
        clim           = [vm_min, vm_max],
        show_scalar_bar= False,
    )

    # Step shoulder markers
    for z_s, d_a, d_b, label in [
        (Z_STEP_1, pin.d_end, pin.d_mid, f"d_end→d_mid\nstep at z={Z_STEP_1*1000:.2f}mm"),
        (Z_STEP_2, pin.d_mid, pin.d_end, f"d_mid→d_end\nstep at z={Z_STEP_2*1000:.2f}mm"),
    ]:
        line = pv.Line((0.0, -D_HOLE * 0.8, z_s), (0.0, D_HOLE * 0.8, z_s))
        pl.add_mesh(line, color="#333333", line_width=2.0)

    # Plate span labels along the Z axis (added as thin horizontal bars)
    for z0, z1, color, label in PLATE_DEFS:
        bar = pv.Line((0.0, B_lug * 0.55, z0), (0.0, B_lug * 0.55, z1))
        pl.add_mesh(bar, color=color, line_width=3.5, opacity=0.85)

    pl.add_text(
        "Side profile  (Y-Z plane, orthographic)\n"
        f"σ_VM: {vm_min:.1f} – {vm_max:.1f} MPa   "
        f"Peak at d_end steps (M_chord couples + d_end < d_mid)",
        position  = "upper_edge",
        font_size = 11,
        color     = "black",
    )
    pl.add_text(
        f"L_pin = {L_PIN*1000:.2f} mm   "
        f"D_hole = {D_HOLE*1000:.2f} mm   "
        f"T_plate = {T_PLATE*1000:.3f} mm   "
        f"H_gap = {H_GAP*1000:.2f} mm",
        position  = "lower_edge",
        font_size = 10,
        color     = "#333333",
    )

    # Orthographic side camera: +X looking toward –X, Z horizontal, Y vertical
    pl.camera_position = _side_camera(
        (scene_cx, scene_cy, scene_cz), scene_r * 1.2
    )
    pl.reset_camera()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    pl.show(screenshot=save, auto_close=True)
    print(f"Wrote {save}")


if __name__ == "__main__":
    main()
