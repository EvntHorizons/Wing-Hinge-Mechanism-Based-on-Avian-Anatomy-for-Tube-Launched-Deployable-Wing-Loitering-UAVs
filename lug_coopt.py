"""
lug_coopt.py
lug plate co-optimization that adds lug width b_lug as a third design variable alongside the pin dimensions d_end and d_mid.
outputs: lug_coopt_analysis.png (trade study figure) and lug_optimized_pin.json (selected compromise design)
"""

import json
import pathlib
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize as pymoo_minimize

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from discreteload_hinge_sim import (
    SteppedPin, MATERIALS, LUG, K_T,
    T_PLATE, D_HOLE,
    V_DESIGN, M_DRAG, V_CRUISE, FS_STATIC, FS_FATIGUE, FS_PROOF,
    V_LAUNCH, M_LAUNCH, V_IMPACT,
    goodman,
    B_LUG_EFF,
)

# lug geometry and mass model
RHO_LUG = 2700.0 # kg/m^3  al 6061:t6
L_LUG = 2.5 * D_HOLE # m 

B_LUG_MIN = 1.5 * D_HOLE #
B_LUG_MAX = 4.0 * D_HOLE #


def lug_mass(B_lug):
    return 2.0 * B_lug * T_PLATE * L_LUG * RHO_LUG


def sigma_net_var(V, B_lug):
    return K_T * V / (max(B_lug - D_HOLE, 1e-12) * T_PLATE)


def sigma_net_B_min(V):
    return D_HOLE + K_T * V * FS_STATIC / (LUG["S_y"] * T_PLATE)


# load case
def load_worst_case():
    p = pathlib.Path(__file__).parent / "mission_optimized_pin.json"
    if p.exists():
        with open(p) as fh:
            d = json.load(fh)
        V_w = float(d["V_worst_N"])
        M_w = float(d["M_chord_worst_Nm"])
        print(f"Mission loads: V = {V_w:.1f} N   M = {M_w:.4f} N·m  "
              f"(mission_optimized_pin.json)")
        return V_w, M_w, d
    print(f"Baseline loads: V = {V_DESIGN:.1f} N   M = {M_DRAG:.4f} N·m  (HINGE)")
    return V_DESIGN, M_DRAG, {}


# optimiser   3:variable: (d_end, d_mid, b_lug)
class LugPinMOO(ElementwiseProblem):
    def __init__(self, material, V_w, M_w, **kw):
        super().__init__(n_var=3, n_obj=2, n_ieq_constr=7,xl=np.array([0.0005, 0.001, B_LUG_MIN]),xu=np.array([D_HOLE, D_HOLE, B_LUG_MAX]),**kw,)
        self.matkey = material
        self.V_w = V_w
        self.M_w = M_w

    def _evaluate(self, x, out, *args, **kw):
        d_end, d_mid, B_lug = x
        if d_end > d_mid:
            out["F"] = [1.0, 1e12]; out["G"] = [1e9] * 7; return
        try:
            pin = SteppedPin(self.matkey, d_end, d_mid)
        except Exception:
            out["F"] = [1.0, 1e12]; out["G"] = [1e9] * 7; return

        mat = pin.mat
        sd = pin.stresses(self.V_w, self.M_w)
        sc = pin.stresses(V_CRUISE, M_DRAG * V_CRUISE / V_DESIGN)
        sa = 0.5 * abs(sd["peak_vm"] - sc["peak_vm"])
        sm = 0.5 * (sd["peak_vm"]  + sc["peak_vm"])
        gr = goodman(sa, sm, mat["S_e"], mat["S_ut"])
        sl = pin.stresses(V_LAUNCH, M_LAUNCH)
        si = pin.stresses(V_IMPACT, M_DRAG)
        sn = sigma_net_var(self.V_w, B_lug) # uses variable b_lug

        out["G"] = [
            sd["peak_vm"]    - mat["S_y"] / FS_STATIC, # g1 pin σ_vm
            sd["bear_PLATE"] - LUG["S_y"] / FS_STATIC, # g2 plate bearing
            sd["bear_wing"]  - LUG["S_y"] / FS_STATIC, # g3 wing bearing
            sn- LUG["S_y"] / FS_STATIC, # g4 net:section (variable b)
            gr- 1.0/ FS_FATIGUE, # g5 goodman
            sl["peak_vm"]- mat["S_y"] / FS_PROOF, # g6 launch proof
            si["peak_vm"]- mat["S_y"] / FS_PROOF, # g7 deploy impact proof
        ]
        m_total = pin.mass() + lug_mass(B_lug)
        vm_worst = max(sd["peak_vm"], sl["peak_vm"], si["peak_vm"])
        out["F"] = [m_total, vm_worst]


def run_lug_optimizer(V_w, M_w, n_gen=100, pop_size=80):
    print(f"\nLug co-optimisztion  B_lug encompasses [{B_LUG_MIN*1000:.1f}, {B_LUG_MAX*1000:.1f}] mm")

    results = {}
    for matkey in MATERIALS:
        prob = LugPinMOO(matkey, V_w, M_w)
        res = pymoo_minimize(prob, NSGA2(pop_size=pop_size), ("n_gen", n_gen), seed=42, verbose=False)
        if res.F is not None and len(res.F) > 0:
            results[matkey] = res
            B_vals = res.X[:, 2] * 1000
            print(f"  {matkey:}: {len(res.F):3d} pts  "
                  f"min total {res.F[:, 0].min()*1000:5.3f} g  "
                  f"B_lug [{B_vals.min():.1f}, {B_vals.max():.1f}] mm")
        else:
            print(f"  {matkey:<14}: no feasible solutions")

    # global compromise
    pts = []
    for mk, res in results.items():
        for x, f in zip(res.X, res.F):
            d_end, d_mid, B_lug = float(x[0]), float(x[1]), float(x[2])
            try:
                p = SteppedPin(mk, d_end, d_mid)
                sd = p.stresses(V_w, M_w)
                sc = p.stresses(V_CRUISE, M_DRAG * V_CRUISE / V_DESIGN)
                sl = p.stresses(V_LAUNCH, M_LAUNCH)
                si = p.stresses(V_IMPACT, M_DRAG)
                sa = 0.5 * abs(sd["peak_vm"] - sc["peak_vm"])
                sm = 0.5 * (sd["peak_vm"]  + sc["peak_vm"])
                gr = goodman(sa, sm, p.mat["S_e"], p.mat["S_ut"])
                sn = sigma_net_var(V_w, B_lug)
                feasible = (
                    sd["peak_vm"]<= p.mat["S_y"] / FS_STATIC and
                    sd["bear_PLATE"] <= LUG["S_y"]/ FS_STATIC and
                    sd["bear_wing"]  <= LUG["S_y"]/ FS_STATIC and
                    sn<= LUG["S_y"]/ FS_STATIC and
                    gr<= 1.0/ FS_FATIGUE and
                    sl["peak_vm"]<= p.mat["S_y"] / FS_PROOF   and
                    si["peak_vm"]<= p.mat["S_y"] / FS_PROOF
                )
                if feasible:
                    pts.append((mk, x, f))
            except Exception:
                pass

    if not pts:
        print("no fully-feasible solutions found.")
        return None, B_LUG_MIN, results

    F_arr = np.array([p[2] for p in pts])
    lo, hi = F_arr.min(0), F_arr.max(0)
    span = np.where(hi > lo, hi - lo, 1.0)
    score = ((F_arr - lo) / span).sum(1)
    bm, bx, _ = pts[int(np.argmin(score))]
    best = SteppedPin(bm, float(bx[0]), float(bx[1]))
    B_best = float(bx[2])

    print(f"\n Selected: {bm}  "f"d_end = {best.d_end*1000:.2f} mm   "f"d_mid = {best.d_mid*1000:.2f} mm   "f"B_lug = {B_best*1000:.2f} mm   "f"total = {(best.mass()+lug_mass(B_best))*1000:.3f} g")
    return best, B_best, results


# visualization 4 panels
try:
    _CMAP = matplotlib.colormaps["tab10"]
except AttributeError:
    _CMAP = plt.cm.get_cmap("tab10") 

MAT_COL = {mk: _CMAP(i) for i, mk in enumerate(MATERIALS)}


def _vm_worst(pin, V_w, M_w):
    sd = pin.stresses(V_w, M_w)
    sl = pin.stresses(V_LAUNCH, M_LAUNCH)
    si = pin.stresses(V_IMPACT, M_DRAG)
    return max(sd["peak_vm"], sl["peak_vm"], si["peak_vm"]) / 1e6


def plot_lug_analysis(results, best_pin, B_best, V_w, M_w, orig_data,save="lug_coopt_analysis.png"):

    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(2, 2, hspace=0.40, wspace=0.34, figure=fig)

    # flatten all pareto points
    all_mass, all_vm, all_Blug, all_mat = [], [], [], []
    for mk, res in results.items():
        for x, f in zip(res.X, res.F):
            all_mass.append(f[0] * 1000)
            all_vm.append(f[1] / 1e6)
            all_Blug.append(x[2] * 1000)
            all_mat.append(mk)
    all_mass = np.array(all_mass)
    all_vm = np.array(all_vm)
    all_Blug = np.array(all_Blug)

    # panel 1 : pareto front coloured by b_lug
    ax1 = fig.add_subplot(gs[0, 0])
    sc1 = ax1.scatter(all_mass, all_vm, c=all_Blug, cmap="plasma",s=20, alpha=0.65, zorder=3)
    cb = fig.colorbar(sc1, ax=ax1)
    cb.set_label("B_lug (mm)", fontsize=9)

    try:
        _o_mat = orig_data.get("material")
        _o_de = orig_data.get("d_end", orig_data.get("d_end_m"))
        _o_dm = orig_data.get("d_mid", orig_data.get("d_mid_m"))
        if _o_mat and _o_de and _o_dm:
            _orig_pin = SteppedPin(_o_mat, float(_o_de), float(_o_dm))
            orig_vm_mpa = _vm_worst(_orig_pin, V_w, M_w)
            orig_mass_g = _orig_pin.mass() * 1000 + lug_mass(B_LUG_EFF) * 1000
            ax1.scatter(orig_mass_g, orig_vm_mpa, marker="s", c="k", s=110, zorder=6,label=f"Pre-lug-opt  B = {B_LUG_EFF*1000:.1f} mm  "f"({orig_mass_g:.2f} g)")
    except Exception:
        pass

    if best_pin is not None:
        bm_g = (best_pin.mass() + lug_mass(B_best)) * 1000
        bv = _vm_worst(best_pin, V_w, M_w)
        ax1.scatter(bm_g, bv, marker="*", c="red", s=200, zorder=7,label=f"Lug-opt   B = {B_best*1000:.1f} mm  ({bm_g:.2f} g)")

    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel("Total System Mass: Pin + Lug Plates (g)")
    ax1.set_ylabel("Worst-case sigma_VM (MPa)")
    ax1.legend(fontsize=8); ax1.grid(True, which="both", alpha=0.25)
    ax1.set_title("Pareto Front: Total Mass vs sigma_VM\n(colored by B_lug)", fontsize=10)

    # panel 2 : b_lug vs total mass, per material
    ax2 = fig.add_subplot(gs[0, 1])
    for mk in MATERIALS:
        mask = np.array(all_mat) == mk
        if mask.any():
            ax2.scatter(all_Blug[mask], all_mass[mask],
                        color=MAT_COL[mk], s=18, alpha=0.55, label=mk)

    ax2.axvline(B_LUG_EFF * 1000, color="k", ls="--", lw=1.5,
                label=f"Original B = {B_LUG_EFF*1000:.1f} mm")
    if best_pin is not None:
        ax2.axvline(B_best * 1000, color="red", ls="-.", lw=1.5,
                    label=f"Selected B = {B_best*1000:.1f} mm")
    ax2.axvline(B_LUG_MIN * 1000, color="grey", ls=":", lw=1.0,
                label=f"B_min (geo) = {B_LUG_MIN*1000:.1f} mm")

    ax2.set_xlabel("Lug Width B_lug (mm)")
    ax2.set_ylabel("Total System Mass (g)")
    ax2.legend(fontsize=7.5, ncol=2); ax2.grid(True, alpha=0.25)
    ax2.set_title("Lug Width vs Total Mass: All Pareto Points by Material", fontsize=10)

    B_struct_min = sigma_net_B_min(V_w)
    ax2.annotate(
        f"Net-section structural min = {B_struct_min*1000:.2f} mm\n" f"is less than geometric min ({B_LUG_MIN*1000:.2f} mm) where constraint never binds\n" f"thus optimizer always selects B = B_min",
        xy=(B_LUG_MIN * 1000, np.percentile(all_mass, 50)), xytext=(B_LUG_MIN * 1000 + 3.5, np.percentile(all_mass, 25)),
        fontsize=7.5, color="grey", arrowprops=dict(arrowstyle="->", color="grey", lw=0.8),
        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="orange", alpha=0.85),
    )

    # panel 3 
    ax3 = fig.add_subplot(gs[1, 0])
    B_range = np.linspace(B_LUG_MIN * 0.8, B_LUG_MAX * 1.1, 500)
    snet_range = np.array([sigma_net_var(V_w, b) / 1e6 for b in B_range])
    ax3.plot(B_range * 1000, snet_range, color="blue", lw=2.2,label="σ_net (analytical)")

    allow_b_mpa = LUG["S_y"] / FS_STATIC / 1e6
    ax3.axhline(allow_b_mpa, color="red", ls="--", lw=1.3, label=f"Allowable = {allow_b_mpa:.0f} MPa (S_y / FS)")

    B_analytical_min = sigma_net_B_min(V_w)
    ax3.axvline(B_analytical_min * 1000, color="darkorange", ls=":", lw=1.3, label=f"Structural B_min = {B_analytical_min*1000:.2f} mm")
    ax3.axvline(B_LUG_MIN * 1000, color="grey", ls=":", lw=1.0, label=f"Geometric B_min = {B_LUG_MIN*1000:.1f} mm")
    ax3.axvline(B_LUG_EFF * 1000, color="k", ls="--", lw=1.3, label=f"Original = {B_LUG_EFF*1000:.1f} mm")
    if best_pin is not None:
        ax3.axvline(B_best * 1000, color="red", ls="-.", lw=1.3, label=f"Selected = {B_best*1000:.1f} mm")

    # shade feasible region
    ax3.fill_betweenx([0, allow_b_mpa], B_LUG_MIN * 1000, B_LUG_MAX * 1000, color="lightgreen", alpha=0.12, label="Feasible B range")

    ax3.set_xlim(B_LUG_MIN * 1000 * 0.8, B_LUG_MAX * 1000 * 1.05)
    ax3.set_ylim(0, allow_b_mpa * 1.4)
    ax3.set_xlabel("Lug Width B_lug (mm)")
    ax3.set_ylabel("Net-Section Stress sigma_net (MPa)")
    ax3.legend(fontsize=7.5); ax3.grid(True, alpha=0.25)
    ax3.set_title("Net-Section Stress vs Lug Width  (analytical, mission loads)", fontsize=10)

    # panel 4: mass breakdown: original vs optimised (stacked bar)
    ax4 = fig.add_subplot(gs[1, 1])

    orig_m_pin = float(orig_data.get("mass_g", 0))
    orig_m_lug = lug_mass(B_LUG_EFF) * 1000

    if best_pin is not None:
        opt_m_pin = best_pin.mass() * 1000
        opt_m_lug = lug_mass(B_best) * 1000
    else:
        opt_m_pin = orig_m_pin
        opt_m_lug = orig_m_lug

    cats = ["Original\n(B = 12.70 mm)", f"Lug-Optimised\n(B = {B_best*1000:.2f} mm)"]
    p_mass = [orig_m_pin, opt_m_pin]
    l_mass = [orig_m_lug, opt_m_lug]
    xp = np.arange(2)
    w = 0.42

    ax4.bar(xp, p_mass, width=w, color="steelblue", label="Pin mass (g)")
    ax4.bar(xp, l_mass, width=w, bottom=p_mass,
            color="darkorange", alpha=0.85, label="Lug plates mass (g)")

    for i, (pm, lm) in enumerate(zip(p_mass, l_mass)):
        total = pm + lm
        ax4.text(xp[i], total + 0.06, f"{total:.3f} g",ha="center", va="bottom", fontsize=10, fontweight="bold")
        if pm > 0.05:
            ax4.text(xp[i], pm / 2, f"{pm:.3f}", ha="center", va="center",fontsize=8, color="white")
        ax4.text(xp[i], pm + lm / 2, f"{lm:.3f}", ha="center", va="center",fontsize=8, color="white")

    savings = (orig_m_pin + orig_m_lug) - (opt_m_pin + opt_m_lug)
    pct = 100 * savings / (orig_m_pin + orig_m_lug) if (orig_m_pin + orig_m_lug) > 0 else 0
    pin_delta = opt_m_pin - orig_m_pin
    lug_delta = opt_m_lug - orig_m_lug

    summary = (f"Total saving: {savings:.3f} g  ({pct:.1f}%)\n"  f"  Pin:  {pin_delta:+.3f} g\n"  f"  Lug:  {lug_delta:+.3f} g")
    ax4.text(0.5, 0.88, summary, transform=ax4.transAxes, ha="center", va="top", fontsize=9, bbox=dict(boxstyle="round", fc="lightyellow", ec="orange", alpha=0.85))

    ax4.set_xticks(xp); ax4.set_xticklabels(cats, fontsize=9)
    ax4.set_ylabel("Mass (g)"); ax4.legend(fontsize=9)
    ax4.set_title("Mass Breakdown: Pin vs Lug Plates", fontsize=10)
    ax4.set_xlim(-0.5, 1.5); ax4.grid(True, axis="y", alpha=0.30)

    # print net-section margins in panel 4
    ms_orig = LUG["S_y"] / FS_STATIC / sigma_net_var(V_w, B_LUG_EFF) - 1.0
    ms_opt = LUG["S_y"] / FS_STATIC / sigma_net_var(V_w, B_best) - 1.0 if B_best else ms_orig
    ax4.text(0.5, 0.38,  f"Net-section margin:\n" f"  Original:+{ms_orig:.2f}\n" f"  Lug-optimised: +{ms_opt:.2f}",
             transform=ax4.transAxes, ha="center", va="top", fontsize=8.5, family="monospace",bbox=dict(boxstyle="round", fc="aliceblue", ec="steelblue", alpha=0.85))

    fig.suptitle(
        f"Lug Plate Co-Optimization -  " f"B_lug encompasses [{B_LUG_MIN*1000:.1f}, {B_LUG_MAX*1000:.1f}] mm  -  "  f"V_worst = {V_w:.1f} N",
        fontsize=12, y=1.003,)
    plt.savefig(save, dpi=150, bbox_inches="tight")
    plt.close(fig)


# json output
def save_lug_results(pin, B_lug, V_w, M_w, save="lug_optimized_pin.json"):
    if pin is None:
        print("No feasible design so it JSON not written.")
        return
    sd = pin.stresses(V_w, M_w)
    sc = pin.stresses(V_CRUISE, M_DRAG * V_CRUISE / V_DESIGN)
    sl = pin.stresses(V_LAUNCH, M_LAUNCH)
    si = pin.stresses(V_IMPACT, M_DRAG)
    sa = 0.5 * abs(sd["peak_vm"] - sc["peak_vm"])
    sm = 0.5 * (sd["peak_vm"]  + sc["peak_vm"])
    gr = goodman(sa, sm, pin.mat["S_e"], pin.mat["S_ut"])
    sn = sigma_net_var(V_w, B_lug)

    out = dict(
        material = pin.matkey,
        d_end_m = float(pin.d_end),
        d_mid_m = float(pin.d_mid),
        B_lug_m = float(B_lug),
        B_lug_original_m = float(B_LUG_EFF),
        mass_pin_g = float(pin.mass() * 1000),
        mass_lug_g = float(lug_mass(B_lug) * 1000),
        mass_total_g = float((pin.mass() + lug_mass(B_lug)) * 1000),
        mass_lug_original_g = float(lug_mass(B_LUG_EFF) * 1000),
        mass_total_original_g= float((pin.mass() + lug_mass(B_LUG_EFF)) * 1000),
        lug_mass_saving_g = float((lug_mass(B_LUG_EFF) - lug_mass(B_lug)) * 1000),
        peak_vm_MPa = float(sd["peak_vm"]    / 1e6),
        bear_PLATE_MPa = float(sd["bear_PLATE"]  / 1e6),
        bear_wing_MPa = float(sd["bear_wing"]   / 1e6),
        sigma_net_MPa = float(sn / 1e6),
        goodman_ratio = float(gr),
        ms_static = float(pin.mat["S_y"] / FS_STATIC / sd["peak_vm"] - 1.0),
        ms_net_section = float(LUG["S_y"] / FS_STATIC / sn - 1.0),
        ms_launch = float(pin.mat["S_y"] / FS_PROOF  / sl["peak_vm"] - 1.0),
        ms_deploy_impact = float(pin.mat["S_y"] / FS_PROOF  / si["peak_vm"] - 1.0),
        ms_goodman = float(1.0 / FS_FATIGUE / gr - 1.0),
        B_lug_structural_min = float(sigma_net_B_min(V_w)),
        B_lug_geometric_min = float(B_LUG_MIN),
        V_design_N = float(V_w),
        M_chord_Nm = float(M_w),
        L_lug_m = float(L_LUG),
    )
    with open(save, "w") as fh:
        json.dump(out, fh, indent=2)


# main
def main():
    V_w, M_w, orig_data = load_worst_case()

    B_net_min = sigma_net_B_min(V_w)
    print(f"\nNet-section structural minimum B_lug = {B_net_min*1000:.3f} mm")
    print(f"Current B_LUG_EFF= {B_LUG_EFF*1000:.1f} mm")
    print(f"Geometric minimum (1.5 × D_HOLE)= {B_LUG_MIN*1000:.1f} mm")
    print(f"Net-section margin at current B_lug   = " f"+{LUG['S_y']/FS_STATIC/sigma_net_var(V_w, B_LUG_EFF)-1:.2f}  " f"(large → lug is over-wide)")

    best_pin, B_best, results = run_lug_optimizer(V_w, M_w, n_gen=100, pop_size=80)

    if best_pin is not None:
        saving = (lug_mass(B_LUG_EFF) - lug_mass(B_best)) * 1000
        print(f"\n  Lug mass saving vs original: {saving:.3f} g")
        print(f"  Total system mass: " f"{(best_pin.mass() + lug_mass(B_best))*1000:.3f} g  " f"(was {(best_pin.mass() + lug_mass(B_LUG_EFF))*1000:.3f} g)")

    plot_lug_analysis(results, best_pin, B_best, V_w, M_w, orig_data)
    save_lug_results(best_pin, B_best, V_w, M_w)

if __name__ == "__main__":
    main()
