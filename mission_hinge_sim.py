"""
mission_hinge_sim.py continuous mission simulation and re-optimisation of the bio-inspired stepped hinge pin through the drone's full flight profile.
outputs: mission_hinge_analysis.png (8=panel structural analysis figure) and mission_optimized_pin.json (re=optimised pin).
"""
import json
import pathlib
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import PchipInterpolator
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize as pymoo_minimize

# import from discreteload_hinge_sim.py
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from discreteload_hinge_sim import (
    SteppedPin, PinMOO,
    MATERIALS, LUG,
    T_PLATE, H_GAP, D_HOLE, L_PIN,
    Z_TOP_PLATE, Z_TOP_WING, Z_BOT_WING, Z_BOT_PLATE,
    Z_STEP_1, Z_STEP_2,
    V_DESIGN, M_DRAG, V_CRUISE, FS_STATIC, FS_FATIGUE, FS_PROOF,
    M_WING_KG, L_HALF_SPAN_M, C_CHORD_M,
    LAUNCH_G, IMPACT_DT, F_STOP, V_LAUNCH, M_LAUNCH, V_IMPACT,
    moments_at, M_combined, sigma_bend, tau_shear,
    sigma_net, goodman,
    B_LUG_EFF, K_T,
)

# mission waypoints
# columns: [t_s, v_ms, alpha_deg, gamma_deg]
MISSION_WP = np.array([
    [-1.0, 0.0, 0.0, 0.00], # 0  stowed (pre-launch)
    [0.0, 15.0, 8.0, 65.00], # 1  tube deployment (+ impact spike)
    [5.0, 35.0, 4.0, 30.00], # 2  powered climb
    [10.0, 60.0, 5.0, 0.00], # 3  cruise
    [15.0, 80.0, 0.0,-50.75], # 4  stoop dive
    [21.0, 55.0, 12.0, 10.00], # 5  10g pull:out  (critical)
    [25.0, 25.0, 5.0, 0.00], # 6  recovery
    [30.0, 45.0, 3.0, 0.00], # 7  final cruise
])
PHASE_LABELS = [
    "Stowed", "Deployment", "Climb", "Cruise",
    "Stoop Dive", "10G Pull-out", "Recovery", "Final Cruise",
]
N_STEPS = 1000

RHO_AIR = 1.225 # kg/m^3
S_HALF = L_HALF_SPAN_M * C_CHORD_M 
AR = 2.0 * L_HALF_SPAN_M / C_CHORD_M 
CD0 = 0.02 
K_DRAG = 1.0 / (np.pi * AR * 0.85) #


def aero_loads_raw(V_ms, alpha_deg, gamma_deg):
    alpha_rad = np.radians(alpha_deg)
    q = 0.5 * RHO_AIR * V_ms**2
    CL = 2.0 * np.pi * np.sin(alpha_rad)
    CD = CD0 + K_DRAG * CL**2
    L_half = q * S_HALF * CL
    D_half = q * S_HALF * CD
    V_root_raw = max(float(L_half), 0.0)
    M_chord_raw = float(D_half) * (C_CHORD_M / 3.0)
    W_wing = M_WING_KG * 9.81
    n_eff = float(L_half) / W_wing if W_wing > 0 else 0.0
    return V_root_raw, M_chord_raw, n_eff


def _compute_aero_scale():
    wp = MISSION_WP[5]
    V_raw, _, _ = aero_loads_raw(float(wp[1]), float(wp[2]), float(wp[3]))
    if V_raw < 1e-6:
        raise ValueError("Pull-out raw shear near zero")
    return V_DESIGN / V_raw


K_AERO = _compute_aero_scale() # dimensionless

# time history generation  (1 000 steps)
def build_time_history(n_steps=N_STEPS):
    """interpolate mission waypoints"""
    t_wp = MISSION_WP[:, 0]
    t_arr = np.linspace(t_wp[0], t_wp[-1], n_steps)

    V_arr = np.clip(PchipInterpolator(t_wp, MISSION_WP[:, 1])(t_arr), 0.0, None)
    alpha_arr = PchipInterpolator(t_wp, MISSION_WP[:, 2])(t_arr)
    gamma_arr = PchipInterpolator(t_wp, MISSION_WP[:, 3])(t_arr)

    V_root_arr = np.zeros(n_steps)
    M_chord_arr = np.zeros(n_steps)
    n_eff_arr = np.zeros(n_steps)
    for i in range(n_steps):
        vr, mc, ne = aero_loads_raw(V_arr[i], alpha_arr[i], gamma_arr[i])
        V_root_arr[i] = K_AERO * vr
        M_chord_arr[i] = K_AERO * mc
        n_eff_arr[i] = ne

    return dict(t=t_arr, V=V_arr, alpha=alpha_arr, gamma=gamma_arr,
                V_root=V_root_arr, M_chord=M_chord_arr, n_eff=n_eff_arr)


# deployment impact spike
def inject_deployment_spike(hist):
    t = hist["t"]
    dt_s = (t[-1] - t[0]) / (len(t) - 1)
    i0 = int(np.argmin(np.abs(t - 0.0)))
    n_pts = max(1, int(np.round(IMPACT_DT / dt_s)))
    i_lo = max(0, i0 - n_pts // 2)
    i_hi = min(len(t), i0 + n_pts // 2 + 1)
    span = i_hi - i_lo
    tri = np.zeros(len(t))
    if span > 0:
        tri[i_lo:i_hi] = np.interp(np.arange(span), [0, span // 2, span - 1], [0.0, 1.0, 0.0])
    hist["V_root"] = hist["V_root"] + F_STOP * tri
    hist["spike_mask"] = tri > 0.0
    hist["i_impact"] = i0
    return hist


# stress evaluation loop
def evaluate_mission_stresses(hist, pin, B_lug=None):
    n = len(hist["t"])
    V_r = hist["V_root"]
    M_c = hist["M_chord"]

    sigma_VM_arr = np.zeros(n)
    sigma_bend_arr = np.zeros(n)
    tau_arr = np.zeros(n)
    bear_plate_arr = np.zeros(n)
    bear_wing_arr = np.zeros(n)
    sigma_net_arr = np.zeros(n)
    goodman_arr = np.zeros(n)

    # cruise baseline for goodman
    sc_base = pin.stresses(V_CRUISE, M_DRAG * V_CRUISE / V_DESIGN)
    sm_base = sc_base["peak_vm"]

    for i in range(n):
        s = pin.stresses(V_r[i], M_c[i])
        sigma_VM_arr[i] = s["peak_vm"]
        bear_plate_arr[i] = s["bear_PLATE"]
        bear_wing_arr[i] = s["bear_wing"]
        if B_lug is not None:
            sigma_net_arr[i] = sigma_net(V_r[i], T_PLATE, B_lug, D_HOLE)
        else:
            sigma_net_arr[i] = s["sigma_net"]
        M_crit = M_combined(L_PIN / 2.0, V_r[i], M_c[i])
        sigma_bend_arr[i] = sigma_bend(M_crit, pin.d_mid)
        tau_arr[i] = tau_shear(V_r[i], pin.d_mid)
        sa_i = 0.5 * abs(s["peak_vm"] - sm_base)
        sm_i = 0.5 * (s["peak_vm"] + sm_base)
        goodman_arr[i] = goodman(sa_i, sm_i, pin.mat["S_e"], pin.mat["S_ut"])

    allow_s = pin.mat["S_y"] / FS_STATIC
    allow_b = LUG["S_y"]  / FS_STATIC
    allow_g = 1.0 / FS_FATIGUE

    return dict(
        sigma_VM = sigma_VM_arr,
        sigma_bend = sigma_bend_arr,
        tau = tau_arr,
        bear_plate = bear_plate_arr,
        bear_wing = bear_wing_arr,
        sigma_net = sigma_net_arr,
        goodman_ratio = goodman_arr,
        MS_static = allow_s / np.maximum(sigma_VM_arr, 1e-12) - 1.0,
        MS_bear_plate = allow_b / np.maximum(bear_plate_arr, 1e-12) - 1.0,
        MS_bear_wing = allow_b / np.maximum(bear_wing_arr, 1e-12) - 1.0,
        MS_net = allow_b / np.maximum(sigma_net_arr, 1e-12) - 1.0,
        MS_goodman = allow_g / np.maximum(goodman_arr, 1e-12) - 1.0,
    )


# worst-case load extraction
def extract_worst_case_loads(hist, stresses):
    """find worst aero timestep (excluding impact spike) and rms fatigue load."""
    spike_mask = hist.get("spike_mask", np.zeros(len(hist["t"]), dtype=bool))
    vm_masked = np.where(spike_mask, 0.0, stresses["sigma_VM"])
    i_worst = int(np.argmax(vm_masked))
    V_w = float(hist["V_root"][i_worst])
    M_w = float(hist["M_chord"][i_worst])
    no_spike = hist["V_root"][~spike_mask]
    V_fat = float(np.sqrt(np.mean(no_spike**2))) if len(no_spike) > 0 else V_CRUISE
    M_fat = M_w * V_fat / V_w if V_w > 0 else 0.0
    return dict(
        i_worst = i_worst,
        t_worst = float(hist["t"][i_worst]),
        V_worst = V_w,
        M_chord_worst = M_w,
        V_fatigue = V_fat,
        M_fatigue = float(M_fat),
    )


# mission nsga-ii re-optimiser
class MissionPinMOO(ElementwiseProblem):
    """nsga-ii problem with mission worst-case and fatigue loads."""

    def __init__(self, material, V_mission, M_mission, V_fat, M_fat, **kw):
        super().__init__(n_var=2, n_obj=2, n_ieq_constr=7, xl=np.array([0.0005, 0.001]), xu=np.array([D_HOLE, D_HOLE]), **kw)
        self.matkey = material
        self.V_m, self.M_m = V_mission, M_mission
        self.V_f, self.M_f = V_fat, M_fat

    def _evaluate(self, x, out, *args, **kw):
        d_end, d_mid = x
        if d_end > d_mid:
            out["F"] = [1.0, 1e12]; out["G"] = [1e9] * 7; return
        try:
            pin = SteppedPin(self.matkey, d_end, d_mid)
        except Exception:
            out["F"] = [1.0, 1e12]; out["G"] = [1e9] * 7; return

        mat = pin.mat
        sd = pin.stresses(self.V_m, self.M_m)
        sc = pin.stresses(self.V_f, self.M_f)
        sa = 0.5 * abs(sd["peak_vm"] - sc["peak_vm"])
        sm = 0.5 * (sd["peak_vm"]  + sc["peak_vm"])
        gr = goodman(sa, sm, mat["S_e"], mat["S_ut"])
        sl = pin.stresses(V_LAUNCH, M_LAUNCH)
        si = pin.stresses(V_IMPACT, M_DRAG) # m_drag

        out["G"] = [
            sd["peak_vm"]  - mat["S_y"] / FS_STATIC, # g1
            sd["bear_PLATE"]- LUG["S_y"] / FS_STATIC, # g2
            sd["bear_wing"] - LUG["S_y"] / FS_STATIC, # g3
            sd["sigma_net"] - LUG["S_y"] / FS_STATIC, # g4
            gr - 1.0 / FS_FATIGUE, # g5
            sl["peak_vm"] - mat["S_y"] / FS_PROOF, # g6
            si["peak_vm"]  - mat["S_y"] / FS_PROOF, # g7
        ]
        vm_worst = max(sd["peak_vm"], sl["peak_vm"], si["peak_vm"])
        out["F"] = [pin.mass(), vm_worst]


def run_mission_optimizer(worst_case, n_gen=80, pop_size=60):
    V_m, M_m = worst_case["V_worst"], worst_case["M_chord_worst"]
    V_f, M_f = worst_case["V_fatigue"], worst_case["M_fatigue"]
    print(f"\nMission NSGA-II  V_mission={V_m:.1f} N  M_mission={M_m:.4f} N·m")

    results = {}
    for matkey in MATERIALS:
        prob = MissionPinMOO(matkey, V_m, M_m, V_f, M_f)
        res = pymoo_minimize(prob, NSGA2(pop_size=pop_size), ("n_gen", n_gen), seed=42, verbose=False)
        if res.F is not None and len(res.F) > 0:
            results[matkey] = res
            print(f"  {matkey:}: {len(res.F):3d} pts, " f"min mass {res.F[:, 0].min()*1000:6.3f} g, " f"min sigma_VM {res.F[:, 1].min()/1e6:5.0f} MPa")
        else:
            print(f"  {matkey:}: no feasible solutions")

    # global compromise: normalised (mass + σ_vm) score : same as run_optimizer()
    pts = []
    for mk, res in results.items():
        for x, f in zip(res.X, res.F):
            try:
                p = SteppedPin(mk, float(x[0]), float(x[1]))
                sd = p.stresses(V_m, M_m)
                sc = p.stresses(V_f, M_f)
                sl = p.stresses(V_LAUNCH, M_LAUNCH)
                si = p.stresses(V_IMPACT, M_DRAG)
                sa = 0.5 * abs(sd["peak_vm"] - sc["peak_vm"])
                sm = 0.5 * (sd["peak_vm"]  + sc["peak_vm"])
                gr = goodman(sa, sm, p.mat["S_e"], p.mat["S_ut"])
                if (sd["peak_vm"]  <= p.mat["S_y"] / FS_STATIC and
                    sd["bear_PLATE"] <= LUG["S_y"]  / FS_STATIC and
                    sd["bear_wing"]  <= LUG["S_y"] / FS_STATIC and
                    sd["sigma_net"]  <= LUG["S_y"] / FS_STATIC and
                    gr  <= 1.0   / FS_FATIGUE and
                    sl["peak_vm"] <= p.mat["S_y"] / FS_PROOF   and
                    si["peak_vm"] <= p.mat["S_y"] / FS_PROOF):
                    pts.append((mk, x, f))
            except Exception:
                pass

    if not pts:
        print("no fully-feasible mission solutions.")
        return None, results

    F_arr = np.array([p[2] for p in pts])
    lo, hi = F_arr.min(0), F_arr.max(0)
    span = np.where(hi > lo, hi - lo, 1.0)
    score = ((F_arr - lo) / span).sum(1)
    bm, bx, _ = pts[int(np.argmin(score))]
    best = SteppedPin(bm, float(bx[0]), float(bx[1]))
    print(f"\n Selected: {bm}  "f"d_end={best.d_end*1000:.2f} mm  d_mid={best.d_mid*1000:.2f} mm  " f"mass={best.mass()*1000:.3f} g")
    return best, results


def run_orig_optimizer(n_gen=80, pop_size=60):
    """original pinmoo per material for pareto comparison (panel 7)."""
    results = {}
    for matkey in MATERIALS:
        prob = PinMOO(matkey)
        res = pymoo_minimize(prob, NSGA2(pop_size=pop_size), ("n_gen", n_gen), seed=42, verbose=False)
        if res.F is not None and len(res.F) > 0:
            results[matkey] = res
            print(f"  {matkey:<14}: {len(res.F):3d} pts")
    return results


# visualization  (8 panels)
try:
    PHASE_CMAP = matplotlib.colormaps["tab10"]
except AttributeError:
    PHASE_CMAP = plt.cm.get_cmap("tab10")


def _phase_index(t_val):
    t_wp = MISSION_WP[:, 0]
    idx = int(np.searchsorted(t_wp, t_val, side="right")) - 1
    return int(np.clip(idx, 0, len(PHASE_LABELS) - 1))


def _add_phase_bands(ax):
    t_wp = MISSION_WP[:, 0]
    trans = ax.get_xaxis_transform()
    for i in range(len(t_wp) - 1):
        ax.axvspan(t_wp[i], t_wp[i + 1], color=PHASE_CMAP(i), alpha=0.07, zorder=0)
        ax.text( (t_wp[i] + t_wp[i + 1]) / 2.0, 0.91,  PHASE_LABELS[i + 1], ha="center", va="top", transform=trans, fontsize=6.5, rotation=40, color=PHASE_CMAP(i), clip_on=True,
        )

def _p1_mission_profile(ax, hist):
    t = hist["t"]
    ax1b = ax.twinx()
    ax.plot(t, hist["V"], color="steelblue", lw=1.8, label="V (m/s)")
    ax.plot(t, hist["n_eff"], color="forestgreen", lw=1.5, label="n_eff (–)")
    ax1b.plot(t, hist["gamma"], color="darkorange", lw=1.2, ls="--", label="γ (deg)")
    ax.scatter(MISSION_WP[:, 0], MISSION_WP[:, 1], color="steelblue", s=40, zorder=5, marker="D")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("V (m/s)  |  n_eff (–)")
    ax1b.set_ylabel("γ (deg)", color="darkorange")
    ax1b.tick_params(axis="y", colors="darkorange")
    lines = ax.get_lines() + ax1b.get_lines()
    ax.legend(lines, [ln.get_label() for ln in lines], fontsize=8, loc="upper right")
    ax.set_title("Mission Profile:  Velocity, Load Factor, Flight-Path Angle", fontsize=10)
    _add_phase_bands(ax)


def _p2_stress_history(ax, hist, stresses, worst_case, pin):
    t = hist["t"]
    i_w = worst_case["i_worst"]
    ax.plot(t, stresses["sigma_VM"]  / 1e6, color="red", lw=2.0, label="sigma_VM (MPa)")
    ax.plot(t, stresses["sigma_bend"] / 1e6, color="blue", lw=1.4, label="sigma_bend (MPa)")
    ax.plot(t, stresses["tau"]  / 1e6, color="green", lw=1.4, label="tau (MPa)")
    allow_s_mpa = pin.mat["S_y"] / FS_STATIC / 1e6
    s_e_mpa = pin.mat["S_e"] / 1e6
    ax.axhline(allow_s_mpa, color="red", ls="--", lw=1.0,  label=f"S_y/FS = {allow_s_mpa:.0f} MPa")
    ax.axhline(s_e_mpa, color="navy", ls=":", lw=1.0, label=f"S_e = {s_e_mpa:.0f} MPa")
    ax.axvline(t[i_w], color="crimson", lw=1.2, ls="-.", alpha=0.7)
    ax.text(t[i_w] + 0.3, 0.93, "worst", transform=ax.get_xaxis_transform(), fontsize=7, color="crimson", va="top", rotation=90)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Stress (MPa)")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Pin Stress History at Critical Section (d_mid)", fontsize=10)
    _add_phase_bands(ax)


def _p3_shear_moment(ax, hist):
    t = hist["t"]
    n = len(t)
    ax3b = ax.twinx()
    M_root_arr = np.array([ M_combined(L_PIN / 2.0, hist["V_root"][i], hist["M_chord"][i]) * 1000.0
        for i in range(n)])
    ax.plot(t, hist["V_root"], color="blue", lw=1.8, label="V_root (N)")
    ax3b.plot(t, M_root_arr, color="darkorange", lw=1.4, label="|M| at midspan (N·mm)")
    i0 = hist["i_impact"]
    ax.annotate(  f"Hard-stop\nF_stop = {F_STOP:.0f} N", xy=(t[i0], hist["V_root"][i0]), xytext=(t[i0] + 2.5, hist["V_root"][i0] * 0.65), arrowprops=dict(arrowstyle="->", color="k", lw=0.8), fontsize=7,)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Shear V_root (N)")
    ax3b.set_ylabel("|M| at midspan (N·mm)", color="darkorange")
    ax3b.tick_params(axis="y", colors="darkorange")
    lines3 = ax.get_lines() + ax3b.get_lines()
    ax.legend(lines3, [ln.get_label() for ln in lines3], fontsize=8)
    ax.set_title("Shear Force and Bending Moment at Hinge Root", fontsize=10)
    _add_phase_bands(ax)


def _p4_bearing(ax, hist, stresses):
    t = hist["t"]
    ax.plot(t, stresses["bear_plate"] / 1e6, color="navy", lw=1.6, label="σ_bear PLATE")
    ax.plot(t, stresses["bear_wing"]  / 1e6, color="purple", lw=1.6, label="σ_bear wing")
    ax.plot(t, stresses["sigma_net"]  / 1e6, color="green", lw=1.4, label="sigma_net (hole K_T)")
    allow_b_mpa = LUG["S_y"] / FS_STATIC / 1e6
    ax.axhline(allow_b_mpa, color="red", ls="--", lw=1.0, label=f"LUG allowable {allow_b_mpa:.0f} MPa")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Stress (MPa)")
    ax.legend(fontsize=8)
    ax.set_title("Bearing and Net-Section Stresses vs Time", fontsize=10)
    _add_phase_bands(ax)


def _p5_margins(ax, hist, stresses):
    t = hist["t"]
    ms_specs = [
        ("MS_static", "red", "MS static (σ_VM)"),
        ("MS_bear_plate", "navy", "MS bear PLATE"),
        ("MS_bear_wing", "purple", "MS bear wing"),
        ("MS_net", "green", "MS net section"),
        ("MS_goodman", "darkorange", "MS Goodman"),
    ]
    for key, col, lbl in ms_specs:
        ms_clipped = np.clip(stresses[key], -2.0, 20.0)
        ax.plot(t, ms_clipped, color=col, lw=1.4, label=lbl)
        ax.fill_between(t, ms_clipped, 0, where=stresses[key] < 0, color=col, alpha=0.12)
    ax.axhline(0, color="k", lw=1.5)
    ax.set_ylim(-0.6, 12)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Margin of Safety")
    ax.legend(fontsize=7.5, ncol=2)
    ax.set_title("Margin of Safety vs Time (all failure modes)", fontsize=10)
    _add_phase_bands(ax)

def _p6_goodman(ax, hist, stresses, worst_case, pin):
    t = hist["t"]
    i_w = worst_case["i_worst"]
    sc_b = pin.stresses(V_CRUISE, M_DRAG * V_CRUISE / V_DESIGN)
    sm_base = sc_b["peak_vm"] / 1e6
    vm_mpa = stresses["sigma_VM"] / 1e6
    sa_arr = 0.5 * np.abs(vm_mpa - sm_base)
    sm_arr = 0.5 * (vm_mpa + sm_base)
    ph_idx = np.array([_phase_index(ti) for ti in t])
    for pi in range(len(PHASE_LABELS)):
        mask = ph_idx == pi
        if mask.any():
            ax.scatter(sm_arr[mask], sa_arr[mask], c=[PHASE_CMAP(pi)],
                       s=6, alpha=0.55, label=PHASE_LABELS[pi])
    ax.scatter(sm_arr[i_w], sa_arr[i_w], marker="*", c="crimson", #love crimson and maroon
               s=150, zorder=6, label="worst case")
    S_e_p = pin.mat["S_e"] / FS_FATIGUE / 1e6
    S_ut_p = pin.mat["S_ut"] / FS_FATIGUE / 1e6
    sm_line = np.linspace(0, S_ut_p, 200)
    ax.plot(sm_line, np.maximum(S_e_p * (1.0 - sm_line / S_ut_p), 0),"r--", lw=1.5, label="Goodman limit")
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    ax.set_xlabel("Mean Stress sigma_m (MPa)"); ax.set_ylabel("Stress Amplitude sigma_a (MPa)")
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("Goodman Diagram for All Mission Timesteps", fontsize=10)
    ax.grid(True, alpha=0.25)

def _p7_pareto(ax, orig_results, mission_results):
    mat_cols = {mk: PHASE_CMAP(i) for i, mk in enumerate(MATERIALS)}
    for mk, res in orig_results.items():
        F = res.F
        ax.scatter(F[:, 0] * 1000, F[:, 1] / 1e6,color=mat_cols.get(mk, "grey"), alpha=0.22, s=14,label=f"{mk} orig")
    for mk, res in mission_results.items():
        F = res.F
        ax.scatter(F[:, 0] * 1000, F[:, 1] / 1e6,
                   color=mat_cols.get(mk, "grey"), alpha=0.75, s=20, marker="^",
                   label=f"{mk} mission")
    ax.set_xlabel("Pin Mass (g)"); ax.set_ylabel("Worst-case sigma_VM (MPa)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.legend(fontsize=6.0, ncol=2)
    ax.set_title("Pareto Front: Original vs Mission-Derived Loads", fontsize=10)
    ax.grid(True, which="both", alpha=0.25)

def _p8_bmd(ax, worst_case, pin):
    z_arr = np.linspace(0, L_PIN, 600)
    V_w = worst_case["V_worst"]
    M_w = worst_case["M_chord_worst"]
    M_y_arr = np.array([moments_at(z, V_w, M_w)[0] * 1000.0 for z in z_arr])
    M_x_arr = np.array([moments_at(z, V_w, M_w)[1] * 1000.0 for z in z_arr])
    M_c_arr = np.array([M_combined(z, V_w, M_w) * 1000.0 for z in z_arr])
    z_mm = z_arr * 1000.0
    ax.plot(z_mm, M_y_arr, color="blue", lw=1.5, label="M_y transverse")
    ax.plot(z_mm, M_x_arr, color="darkorange", lw=1.5, label="M_x chordwise")
    ax.plot(z_mm, M_c_arr, color="red", lw=2.0, label="|M| combined")
    for z_b, lbl in zip(
        [Z_TOP_PLATE * 1000, Z_TOP_WING * 1000,
         Z_BOT_WING  * 1000, Z_BOT_PLATE * 1000],
        ["top plate", "top wing", "bot wing", "bot plate"],
    ):
        ax.axvline(z_b, color="grey", ls="--", lw=0.8)
        ax.text(z_b + 0.1, 0.02, lbl, transform=ax.get_xaxis_transform(), fontsize=6, rotation=90, va="bottom", color="grey")
    ax.axvspan(0, Z_STEP_1 * 1000, color="skyblue", alpha=0.15,label=f"d_end = {pin.d_end*1000:.2f} mm")
    ax.axvspan(Z_STEP_2 * 1000, L_PIN    * 1000, color="skyblue", alpha=0.15)
    ax.axvspan(Z_STEP_1 * 1000, Z_STEP_2 * 1000, color="lightcoral", alpha=0.12, label=f"d_mid = {pin.d_mid*1000:.2f} mm")
    ax.set_xlabel("z along pin (mm)"); ax.set_ylabel("Moment (N*mm)")
    ax.legend(fontsize=8)
    ax.set_title(f"Bending Moment Diagram (Worst Case)  "f"(t = {worst_case['t_worst']:.1f} s,  V = {V_w:.1f} N)", fontsize=10)
    ax.grid(True, alpha=0.25)


# 8 figure panel AND 8 individual figures for report pasting

def plot_mission_analysis(hist, stresses, worst_case, pin, orig_results, mission_results, save="mission_hinge_analysis.png"):
    fig = plt.figure(figsize=(22, 28))
    gs = gridspec.GridSpec( 4, 2, figure=fig,top=0.963, bottom=0.038, left=0.07, right=0.95, hspace=0.62, wspace=0.42,
    )

    _p1_mission_profile(fig.add_subplot(gs[0, 0]), hist)
    _p2_stress_history( fig.add_subplot(gs[0, 1]), hist, stresses, worst_case, pin)
    _p3_shear_moment(   fig.add_subplot(gs[1, 0]), hist)
    _p4_bearing( fig.add_subplot(gs[1, 1]), hist, stresses)
    _p5_margins(fig.add_subplot(gs[2, 0]), hist, stresses)
    _p6_goodman(fig.add_subplot(gs[2, 1]), hist, stresses, worst_case, pin)
    _p7_pareto(fig.add_subplot(gs[3, 0]), orig_results, mission_results)
    _p8_bmd(fig.add_subplot(gs[3, 1]), worst_case, pin)

    fig.suptitle(f"Mission Hinge Structural Analysis: {pin.matkey}  " f"d_end = {pin.d_end*1000:.2f} mm   d_mid = {pin.d_mid*1000:.2f} mm   "
        f"mass = {pin.mass()*1000:.3f} g",  fontsize=13,
    )
    plt.savefig(save, dpi=150, bbox_inches="tight")
    plt.close(fig)



# individual panel exports 

_PANEL_SPECS = [
    ("01_mission_profile", (14, 7), lambda ax, kw: _p1_mission_profile(ax, kw["hist"])), 
    ("02_stress_history", (14, 7), lambda ax, kw: _p2_stress_history(ax, kw["hist"], kw["stresses"], kw["worst_case"], kw["pin"])),
    ("03_shear_moment", (14, 7), lambda ax, kw: _p3_shear_moment(ax, kw["hist"])), 
    ("04_bearing_stresses", (14, 7), lambda ax, kw: _p4_bearing(ax, kw["hist"], kw["stresses"])), 
    ("05_margins_of_safety", (14, 7), lambda ax, kw: _p5_margins(ax, kw["hist"], kw["stresses"])), 
    ("06_goodman_diagram", (10, 7), lambda ax, kw: _p6_goodman(ax, kw["hist"], kw["stresses"], kw["worst_case"], kw["pin"])), 
    ("07_pareto_comparison", (10, 7), lambda ax, kw: _p7_pareto(ax, kw["orig_results"], kw["mission_results"])),
    ("08_bmd_worst_case", (10, 7),lambda ax, kw: _p8_bmd(ax, kw["worst_case"], kw["pin"])),
]

def save_individual_panels(hist, stresses, worst_case, pin,  orig_results, mission_results, outdir="mission_panels"):
    out = pathlib.Path(__file__).parent / outdir
    out.mkdir(exist_ok=True)
    kw = dict(hist=hist, stresses=stresses, worst_case=worst_case, pin=pin, orig_results=orig_results, mission_results=mission_results)
    subtitle = (f"{pin.matkey}  ; d_end = {pin.d_end*1000:.2f} mm  "  f"d_mid = {pin.d_mid*1000:.2f} mm  ;  mass = {pin.mass()*1000:.3f} g")

    for name, figsize, draw_fn in _PANEL_SPECS:
        fig, ax = plt.subplots(figsize=figsize)
        draw_fn(ax, kw)
        fig.suptitle(subtitle, fontsize=8, color="grey", y=1.01)
        fig.tight_layout()
        fpath = out / f"{name}.png"
        plt.savefig(fpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"{fpath.name}")

# json output for data processing ggg
def save_mission_results(pin, worst_case, stresses, save="mission_optimized_pin.json"):
    V_w, M_w = worst_case["V_worst"], worst_case["M_chord_worst"]
    V_f, M_f = worst_case["V_fatigue"], worst_case["M_fatigue"]
    sd = pin.stresses(V_w, M_w)
    sc = pin.stresses(V_f, M_f)
    sl = pin.stresses(V_LAUNCH, M_LAUNCH)
    si = pin.stresses(V_IMPACT, M_DRAG)
    sa = 0.5 * abs(sd["peak_vm"] - sc["peak_vm"])
    sm = 0.5 * (sd["peak_vm"]  + sc["peak_vm"])
    gr = goodman(sa, sm, pin.mat["S_e"], pin.mat["S_ut"])

    out = dict(
        material = pin.matkey,
        d_end = float(pin.d_end),
        d_mid = float(pin.d_mid),
        mass_g = float(pin.mass() * 1000),
        V_worst_N = float(V_w),
        M_chord_worst_Nm = float(M_w),
        t_worst_s = float(worst_case["t_worst"]),
        V_fatigue_mean_N = float(V_f),
        M_fatigue_mean_Nm = float(M_f),
        peak_vm_worst_MPa = float(sd["peak_vm"]/ 1e6),
        bear_PLATE_MPa = float(sd["bear_PLATE"]  / 1e6),
        bear_wing_MPa = float(sd["bear_wing"]/ 1e6),
        sigma_net_MPa = float(sd["sigma_net"]/ 1e6),
        goodman_ratio = float(gr),
        ms_static = float(pin.mat["S_y"] / FS_STATIC / sd["peak_vm"] - 1.0),
        ms_launch = float(pin.mat["S_y"] / FS_PROOF  / sl["peak_vm"] - 1.0),
        ms_deploy_impact = float(pin.mat["S_y"] / FS_PROOF  / si["peak_vm"] - 1.0),
        ms_goodman = float(1.0 / FS_FATIGUE / gr - 1.0),
        min_MS_static = float(np.min(stresses["MS_static"])),
        min_MS_bear_plate = float(np.min(stresses["MS_bear_plate"])),
        min_MS_bear_wing = float(np.min(stresses["MS_bear_wing"])),
        min_MS_net = float(np.min(stresses["MS_net"])),
        min_MS_goodman = float(np.min(stresses["MS_goodman"])),
        K_AERO = float(K_AERO),
        F_stop_N = float(F_STOP),
        V_impact_N = float(V_IMPACT),
        launch_g = float(LAUNCH_G),
        impact_dt_s = float(IMPACT_DT),
        t_plate_m = float(T_PLATE),
        h_gap_m = float(H_GAP),
        d_hole_m = float(D_HOLE),
        L_pin_m = float(L_PIN),
    )
    with open(save, "w") as fh:
        json.dump(out, fh, indent=2)


# main
def _load_base_pin(json_name):
    """load a steppedpin from a json file; handles d_end / d_end_m key variants."""
    p = pathlib.Path(__file__).parent / json_name
    if not p.exists():
        raise FileNotFoundError(f"{json_name} not found in {p.parent}")
    with open(p) as fh:
        d = json.load(fh)
    mat = d["material"]
    d_end = float(d.get("d_end", d.get("d_end_m")))
    d_mid = float(d.get("d_mid", d.get("d_mid_m")))
    return SteppedPin(mat, d_end, d_mid), d


def main(pin_json = "optimized_pin.json",out_fig = "mission_hinge_analysis.png",panels_dir = "mission_panels", result_json = "mission_optimized_pin.json"):
    # load input pin
    try:
        base_pin, pin_data = _load_base_pin(pin_json)
    except FileNotFoundError:
        if pin_json == "optimized_pin.json":
            print("optimized_pin.json not found — running HINGE optimizer first …")
            from discreteload_hinge_sim import run_optimizer as hinge_run
            base_pin = hinge_run()
            base_pin, pin_data = _load_base_pin(pin_json)
        else:
            raise

    B_lug = pin_data.get("B_lug_m")
    if B_lug is not None:
        print(f"  Lug width from JSON: B_lug = {B_lug*1000:.2f} mm  "
              f"(vs B_LUG_EFF = {B_LUG_EFF*1000:.2f} mm)")

    print(f"Loaded pin ({pin_json}): {base_pin.matkey}  "
          f"d_end={base_pin.d_end*1000:.2f} mm  d_mid={base_pin.d_mid*1000:.2f} mm")
    print(f"\nK_AERO = {K_AERO:.4f}  (aero normalisation factor at pull-out)")
    print(f"F_STOP = {F_STOP:.1f} N  ({F_STOP/V_DESIGN:.1f}× V_DESIGN)")

    hist = build_time_history(N_STEPS)
    hist = inject_deployment_spike(hist)
    i_pk = int(hist["V_root"].argmax())
    print(f"  Peak V_root (incl. spike): {hist['V_root'].max():.1f} N  "
          f"at t = {hist['t'][i_pk]:.3f} s")

    # evaluate base pin over mission
    print("Evaluating stresses on base pin")
    stresses = evaluate_mission_stresses(hist, base_pin, B_lug=B_lug)
    worst_case = extract_worst_case_loads(hist, stresses)
    print(f"Worst aero: t = {worst_case['t_worst']:.2f} s  " f"V = {worst_case['V_worst']:.1f} N  " f"M = {worst_case['M_chord_worst']:.4f} N·m")

    # mission re-optimisation
    mission_pin, mission_results = run_mission_optimizer(worst_case, n_gen=80, pop_size=60)

    # original optimisation for pareto comparison
    orig_results = run_orig_optimizer(n_gen=80, pop_size=60)

    # final evaluation on mission pin (same b_lug applies)
    print("\nEvaluating stresses on mission-optimised pin")
    mission_stresses = evaluate_mission_stresses(hist, mission_pin, B_lug=B_lug)
    mission_worst_case = extract_worst_case_loads(hist, mission_stresses)

    print("\n Maximum stresses over full mission (MPa):")
    for key in ("sigma_VM", "sigma_bend", "tau", "bear_plate", "bear_wing", "sigma_net"):
        print(f" {key:}: {np.max(mission_stresses[key]) / 1e6:.2f}")

    print("\n  Minimum margins of safety over full mission:")
    for key in ("MS_static", "MS_bear_plate", "MS_bear_wing", "MS_net", "MS_goodman"):
        print(f" {key:}: {np.min(mission_stresses[key]):+.4f}")

    plot_mission_analysis(hist = hist,stresses = mission_stresses,worst_case = mission_worst_case,pin = mission_pin,orig_results = orig_results,
        mission_results = mission_results,save = out_fig,
    )

    save_individual_panels(hist = hist,stresses = mission_stresses,worst_case = mission_worst_case,
        pin = mission_pin,orig_results = orig_results,     mission_results = mission_results,outdir = panels_dir,
    )

    save_mission_results( pin = mission_pin, worst_case = mission_worst_case,stresses = mission_stresses,save = result_json,)

    print(f"{out_fig}")
    print(f"  {panels_dir}/  (8 individual panels)")
    print(f"{result_json}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Mission hinge structural analysis")
    ap.add_argument("--pin-json", default="optimized_pin.json",help="Input pin JSON (default: optimized_pin.json)")
    ap.add_argument("--out-fig", default="mission_hinge_analysis.png",help="Combined 8-panel output figure")
    ap.add_argument("--panels-dir", default="mission_panels",help="Subfolder for individual panel PNGs")
    ap.add_argument("--result-json", default="mission_optimized_pin.json",help="Output JSON with re-optimised pin results")
    args = ap.parse_args()
    main(pin_json = args.pin_json,
         out_fig = args.out_fig,
         panels_dir = args.panels_dir,
         result_json = args.result_json)