"""
hinge_pin.py

bio:inspired stepped pin for a tube:launched munition wing hinge.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize

# geometry (from cad)
T_PLATE = 0.125 * 0.0254 # 3.175 mm plate thickness
H_GAP = 0.250 * 0.0254 # 6.350 mm, gap between the two wing plates
D_HOLE = 0.250 * 0.0254 # 6.350 mm, pin hole diameter
B_LUG_EFF = 0.500 * 0.0254 # 12.70 mm, effective lug width (net:section)

L_PIN = 4.0 * T_PLATE + H_GAP # 19.050 mm, total pin length

# plate centers along the pin axis (where the bearing forces act)
Z_TOP_PLATE = T_PLATE / 2.0 # 1.5875 mm support 1
Z_TOP_WING = 1.5 * T_PLATE # 4.7625 mm  load 1
Z_BOT_WING = 2.5 * T_PLATE + H_GAP # 14.2875 mm load 2
Z_BOT_PLATE = 3.5 * T_PLATE + H_GAP # 17.4625 mm support 2

Z_STEP_1 = T_PLATE # 3.175 mm
Z_STEP_2 = L_PIN - T_PLATE # 15.875 mm
L_END = T_PLATE # length of each d_end region
L_MID = L_PIN - 2.0 * T_PLATE # length of d_mid region (= 2t + h)

# materials
MATERIALS = {
    "Ti-6Al-4V": dict(E=114e9, S_y=900e6, S_ut=950e6, S_e=510e6, rho=4430),
    "17-7PH": dict(E=200e9, S_y=1300e6, S_ut=1450e6, S_e=700e6, rho=7800),
    "CFRP rod": dict(E=130e9, S_y=1500e6, S_ut=1500e6, S_e=700e6, rho=1600),
    "Al 7075-T6": dict(E=71.7e9, S_y=503e6, S_ut=572e6, S_e=159e6, rho=2810),
    "PEEK-CF": dict(E=14e9, S_y=140e6, S_ut=160e6, S_e=55e6, rho=1400),
}
LUG = dict(name="Al 6061-T6", S_y=276e6, S_ut=310e6)
K_T = 2.7 # stress:concentration factor at a circular hole

# load envelope  (worst  case 10g pullup, from prior schrenk analysis)
V_DESIGN = 78.0 # n: peak transverse shear at wing root
M_DRAG = 0.75 # n·m: chordwise bending moment at wing root
V_CRUISE = 30.0 # n: cyclic mean for goodman fatigue

FS_STATIC = 1.5
FS_FATIGUE = 1.2

# dynamic load parameters  (launch + deployment impact)
M_WING_KG = 0.080 # kg 
L_HALF_SPAN_M = 9.0  * 0.0254 
C_CHORD_M = 6.0  * 0.0254

I_WING = M_WING_KG * (L_HALF_SPAN_M**2 / 3.0 + C_CHORD_M**2 / 12.0)

# launch
LAUNCH_G = 20.0 # g 
LAUNCH_MOMENT_ARM = C_CHORD_M / 2.0 # m

# deployment spring
K_SPRING = 1.8 # n·m/rad
THETA_PRE = np.radians(70.0) # rad
THETA_DEPLOY = np.radians(90.0) # rad 

# hard stop
R_STOP = 0.020 # m  
IMPACT_DT = 0.004 # s
COR = 0.30

# proof factor for single event dynamic loads (not cyclic fatigue)
FS_PROOF = 1.25

# pre:compute impact force
_t_eff = min(THETA_DEPLOY, THETA_PRE)
_W_spr = K_SPRING * (THETA_PRE * _t_eff - 0.5 * _t_eff**2) # J
_omega = np.sqrt(2.0 * _W_spr / I_WING) # rad/s 
F_STOP = I_WING * _omega * (1.0 + COR) / IMPACT_DT / R_STOP # N

V_LAUNCH = M_WING_KG * LAUNCH_G * 9.81 # n 
M_LAUNCH = V_LAUNCH * LAUNCH_MOMENT_ARM # n·m
V_IMPACT = V_DESIGN + F_STOP # n

# stress functions  (round cross-section)
def section_modulus(d): return np.pi * d**3 / 32.0
def section_area(d): return np.pi * d**2 / 4.0 
def sigma_bend(M, d): return M / section_modulus(d)
def tau_shear(V, d): return 4.0 * V / (3.0 * section_area(d))
def vm(s, t): return np.sqrt(s * s + 3.0 * t * t)
def sigma_bear(P, t, d): return P / (t * d)
def sigma_net(P, t, b, d):
    return K_T * P / (max(b - d, 1e-9) * t)
def goodman(sa, sm, S_e, S_ut): return sa / S_e + sm / S_ut

# bending moment along the pin
def moments_at(z, V, M_chord):
    P_w = V / 2.0
    # m_y
    if z <= Z_TOP_PLATE or z >= Z_BOT_PLATE:
        M_y = 0.0
    elif z <= Z_TOP_WING:
        M_y = P_w * (z - Z_TOP_PLATE)
    elif z <= Z_BOT_WING:
        M_y = P_w * T_PLATE
    else:
        M_y = P_w * (Z_BOT_PLATE - z)
    # m_x
    if z <= Z_TOP_PLATE or z >= Z_BOT_PLATE:
        M_x = 0.0
    elif z <= Z_TOP_WING:
        M_x = -M_chord / 2.0
    elif z <= Z_BOT_WING:
        M_x = 0.0
    else:
        M_x = +M_chord / 2.0
    return M_y, M_x

def M_combined(z, V, M_chord):
    M_y, M_x = moments_at(z, V, M_chord)
    return np.hypot(M_y, M_x)

# steppedpin
class SteppedPin:
    def __init__(self, material, d_end, d_mid):
        if d_mid > D_HOLE: raise ValueError("d_mid > d_hole")
        if d_end > d_mid: raise ValueError("d_end > d_mid")
        if d_end <= 0: raise ValueError("d_end ≤ 0")
        self.matkey = material
        self.mat = MATERIALS[material]
        self.d_end = d_end
        self.d_mid = d_mid

    def mass(self):
        return self.mat["rho"] * (
            2.0 * section_area(self.d_end) * L_END
                + section_area(self.d_mid) * L_MID
        )

    def diameter_at(self, z):
        return self.d_end if (z <= Z_STEP_1 or z >= Z_STEP_2) else self.d_mid

    def stresses(self, V, M_chord=0.0):
        z_samples = (Z_TOP_PLATE, Z_STEP_1, Z_TOP_WING - 1e-9, Z_TOP_WING + 1e-9, L_PIN / 2.0, Z_BOT_WING - 1e-9, Z_BOT_WING + 1e-9, Z_STEP_2, Z_BOT_PLATE,)
        max_vm = 0.0
        for z in z_samples:
            M = M_combined(z, V, M_chord)
            d = self.diameter_at(z)
            sb = sigma_bend(M, d)
            tv = tau_shear(V, d)
            vmz = vm(sb, tv)
            if vmz > max_vm:
                max_vm = vmz

        # bearing on each plate (v/2 per plate by symmetry)
        bear_PLATE = sigma_bear(V / 2.0, T_PLATE, self.d_end) # outer/plate plates
        bear_wing = sigma_bear(V / 2.0, T_PLATE, self.d_mid) # inner/wing plates
        s_net = sigma_net(V, T_PLATE, B_LUG_EFF, D_HOLE)

        return dict(peak_vm=max_vm, bear_PLATE=bear_PLATE,bear_wing=bear_wing, sigma_net=s_net)

    def evaluate(self):
        s_d = self.stresses(V_DESIGN, M_DRAG)
        s_c = self.stresses(V_CRUISE, M_DRAG * V_CRUISE / V_DESIGN)
        sa = 0.5 * (s_d["peak_vm"] - s_c["peak_vm"])
        sm = 0.5 * (s_d["peak_vm"] + s_c["peak_vm"])
        gr = goodman(sa, sm, self.mat["S_e"], self.mat["S_ut"])
        return dict(s_design=s_d, s_cruise=s_c,peak_vm=s_d["peak_vm"], goodman_ratio=gr,mass=self.mass())

    def predict_life(self):
        results = self.evaluate()
        sa = 0.5 * (results["s_design"]["peak_vm"] - results["s_cruise"]["peak_vm"])
        sm = 0.5 * (results["s_design"]["peak_vm"] + results["s_cruise"]["peak_vm"])
        s_ut = self.mat["S_ut"]
        s_e = self.mat["S_e"]

        # convert to stress amplitude
        if sm >= s_ut: return 0.0
        sigma_ar = sa / (1.0 - (sm / s_ut))

        # check for infinite life
        if sigma_ar <= s_e:
            return float('inf')

        # basquin equation
        A = 0.9 * s_ut
        b = np.log10(s_e / A) / 6.0

        # calc number of cycles
        n_f = (sigma_ar / A)**(1.0 / b)
        print(f"  Number of Cycles: {n_f}")
        return float(n_f)

# optimizer
#     objectives : (mass, worst:case σ_vm across all load cases)
#     constraints: 5 aero/fatigue  +  2 dynamic proof loads  = 7 total
class PinMOO(ElementwiseProblem):
    """
    constraint set
    g1 pin σ_vm  ≤ s_y / fs_static (aero design, 10g pull:up)
    g2  bear_plate ≤ s_y_lug / fs_static (outer plate bearing)
    g3  bear_wing ≤ s_y_lug / fs_static (inner plate bearing)
    g4  σ_net  ≤ s_y_lug / fs_static(plate net:section)
    g5  goodman  ≤ 1 / fs_fatigue (cyclic fatigue)
    g6  σ_vm_launch ≤ s_y / fs_proof (launch proof load)
    g7  σ_vm_impact ≤ s_y / fs_proof deployment impact proof)
    """
    def __init__(self, material, **kw):
        super().__init__(n_var=2, n_obj=2, n_ieq_constr=7,xl=np.array([0.0005, 0.001]), xu=np.array([D_HOLE, D_HOLE]), **kw)
        self.matkey = material

    def _evaluate(self, x, out, *args, **kw):
        d_end, d_mid = x
        if d_end > d_mid:
            out["F"] = [1.0, 1e12]; out["G"] = [1e9] * 7; return
        try:
            pin = SteppedPin(self.matkey, d_end, d_mid)
            r = pin.evaluate()
        except Exception:
            out["F"] = [1.0, 1e12]; out["G"] = [1e9] * 7; return

        sd, mat = r["s_design"], pin.mat

        # aero / fatigue constraints
        g1 = sd["peak_vm"] - mat["S_y"] / FS_STATIC
        g2 = sd["bear_PLATE"] - LUG["S_y"] / FS_STATIC
        g3 = sd["bear_wing"] - LUG["S_y"] / FS_STATIC
        g4 = sd["sigma_net"]  - LUG["S_y"] / FS_STATIC
        g5 = r["goodman_ratio"] - 1.0 / FS_FATIGUE

        # dynamic proof load constraints
        s_launch = pin.stresses(V=V_LAUNCH, M_chord=M_LAUNCH)
        s_impact = pin.stresses(V=V_IMPACT, M_chord=M_DRAG)
        g6 = s_launch["peak_vm"] - mat["S_y"] / FS_PROOF # launch
        g7 = s_impact["peak_vm"] - mat["S_y"] / FS_PROOF # deployment impact

        vm_worst = max(r["peak_vm"], s_launch["peak_vm"], s_impact["peak_vm"])
        out["F"] = [pin.mass(), vm_worst]
        out["G"] = [g1, g2, g3, g4, g5, g6, g7]


def run_optimizer(n_gen=80, pop_size=60):
    print("Bio-Inspired stepped pin (4-plate/lug hinge); NSGA-II per material")
    print(f"Constraints: aero 10g + fatigue + launch {LAUNCH_G:.0f}g + deploy impact " f"(F_stop={F_STOP:.0f} N = {F_STOP/V_DESIGN:.1f}× aero)\n")
    results = {}
    for matkey in MATERIALS:
        prob = PinMOO(matkey)
        res = minimize(prob, NSGA2(pop_size=pop_size),
                        ("n_gen", n_gen), seed=42, verbose=False)
        if res.F is None or len(res.F) == 0:
            print(f"  {matkey:<14}: no feasible solutions"); continue
        results[matkey] = res
        print(f"  {matkey:}: {len(res.F):3d} pts, "
              f"min mass {res.F[:,0].min()*1000:6.3f} g, "
              f"min σ_VM* {res.F[:,1].min()/1e6:5.0f} MPa")

    print("  (* worst-case σ_VM across aero + launch + deployment impact)")

    if not results:
        print("\nNo feasible solutions over any material.")
        return None
    # global compromise smallest normalised 
    pts = []
    for mk, res in results.items():
        for x, f in zip(res.X, res.F):
            try:
                p = SteppedPin(mk, float(x[0]), float(x[1]))
                ev = p.evaluate()
                sd = ev["s_design"]
                sl = p.stresses(V=V_LAUNCH, M_chord=M_LAUNCH)
                si = p.stresses(V=V_IMPACT, M_chord=M_DRAG)
                feasible = (
                    sd["peak_vm"] <= p.mat["S_y"] / FS_STATIC  and
                    sd["bear_PLATE"]<= LUG["S_y"]  / FS_STATIC  and
                    sd["bear_wing"]  <= LUG["S_y"]  / FS_STATIC  and
                    sd["sigma_net"] <= LUG["S_y"]  / FS_STATIC  and
                    ev["goodman_ratio"] <= 1.0 / FS_FATIGUE and
                    sl["peak_vm"]  <= p.mat["S_y"] / FS_PROOF   and
                    si["peak_vm"]   <= p.mat["S_y"] / FS_PROOF
                )
                if feasible:
                    pts.append((mk, x, f))
            except Exception:
                pass

    if not pts:
        print("\nNo fully-feasible solutions after constraint verification.")
        return None

    F = np.array([p[2] for p in pts])
    lo, hi = F.min(0), F.max(0)
    span = np.where(hi > lo, hi - lo, 1.0)
    score = ((F - lo) / span).sum(1)
    bm, bx, bf = pts[int(np.argmin(score))]
    pin = SteppedPin(bm, float(bx[0]), float(bx[1]))
    r = pin.evaluate()

    # dynamic margins for the selected design
    s_launch = pin.stresses(V=V_LAUNCH, M_chord=M_LAUNCH)
    s_impact = pin.stresses(V=V_IMPACT, M_chord=M_DRAG)
    ms_aero = pin.mat["S_y"] / FS_STATIC / r["s_design"]["peak_vm"] - 1.0
    ms_launch = pin.mat["S_y"] / FS_PROOF  / s_launch["peak_vm"] - 1.0
    ms_impact = pin.mat["S_y"] / FS_PROOF  / s_impact["peak_vm"] - 1.0

    vm_map = {
        "aero 10g": (r["s_design"]["peak_vm"], FS_STATIC), f"launch {LAUNCH_G:.0f}g": (s_launch["peak_vm"], FS_PROOF), "deploy impact": (s_impact["peak_vm"], FS_PROOF),}
    governing = max(vm_map, key=lambda k: vm_map[k][0])
    gov_vm, gov_fs = vm_map[governing]

    print(f"\nGlobal compromise design:")
    print(f"Material: {bm}")
    print(f"d_end: {bx[0]*1000:6.3f} mm")
    print(f"d_mid: {bx[1]*1000:6.3f} mm   " f"(d_end / d_mid = {bx[0]/bx[1]:.3f})")
    print(f"Mass: {bf[0]*1000:6.3f} g")
    print(f"Peak sigma_VM*: {bf[1]/1e6:6.1f} MPa  ← governs: {governing}" f" (allow {pin.mat['S_y']/gov_fs/1e6:.0f} MPa, FS={gov_fs})")
    print(f"Goodman ratio: {r['goodman_ratio']:6.3f} " f"(limit {1/FS_FATIGUE:.3f})")
    print(f"Aero MS: {ms_aero:+.3f}  " f"{'works' if ms_aero   >= 0 else 'fails'}")
    print(f"Launch MS: {ms_launch:+.3f}  " f"{'works' if ms_launch >= 0 else 'fails'}")
    print(f"Deploy impact MS: {ms_impact:+.3f}  " f"{'works' if ms_impact >= 0 else 'fails'}")

    out_data = dict(
        material=bm, d_end=float(bx[0]), d_mid=float(bx[1]),
        mass_g=float(bf[0]*1000),
        peak_vm_worst_MPa=float(bf[1]/1e6),
        goodman_ratio=float(r["goodman_ratio"]),
        bear_PLATE_MPa=float(r["s_design"]["bear_PLATE"]/1e6),
        bear_wing_MPa=float(r["s_design"]["bear_wing"]/1e6),
        sigma_net_MPa=float(r["s_design"]["sigma_net"]/1e6),
        ms_launch=float(ms_launch),
        ms_deploy_impact=float(ms_impact),
        # dynamic parameters used
        launch_g=LAUNCH_G, F_stop_N=float(F_STOP),
        V_impact_N=float(V_IMPACT), impact_dt_s=IMPACT_DT,
        # geometry constants
        t_plate=T_PLATE, h_gap=H_GAP, d_hole=D_HOLE, L_pin=L_PIN,
        z_top_PLATE=Z_TOP_PLATE, z_top_wing=Z_TOP_WING,
        z_bot_wing=Z_BOT_WING, z_bot_PLATE=Z_BOT_PLATE,
    )
    with open("optimized_pin.json", "w") as f: json.dump(out_data, f, indent=2)
    plot_design_summary(results, pin, bf)
    return pin

# plotting (single 4-panel summary figure)
def plot_design_summary(results, pin, best_F):
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.25)
    cmap = plt.get_cmap("tab10")

    # subplot a pareto fronts
    ax = fig.add_subplot(gs[0, 0])
    for i, (mk, res) in enumerate(results.items()):
        ax.scatter(res.F[:, 0]*1000, res.F[:, 1]/1e6,
                   s=22, alpha=0.6, color=cmap(i), label=mk)
    ax.scatter(best_F[0]*1000, best_F[1]/1e6, marker="*", s=320,
               edgecolor="k", facecolor="red", linewidth=1.4,
               zorder=10, label="Selected")
    ax.set_xlabel("Pin mass (g)")
    ax.set_ylabel("Worst-case σ_VM (MPa)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    ax.set_title("Pareto fronts: mass vs worst-case σ_VM\n"
                 "(Aero, Launch & Deploy Impact)")

    # subplot b pin profile + plate stack
    ax = fig.add_subplot(gs[0, 1])
    z_pts = np.array([0, Z_STEP_1, Z_STEP_1, Z_STEP_2,
                      Z_STEP_2, L_PIN]) * 1000
    r_pts = np.array([pin.d_end, pin.d_end, pin.d_mid, pin.d_mid,
                      pin.d_end, pin.d_end]) / 2.0 * 1000
    ax.fill_between(z_pts, -r_pts, +r_pts, color="darkorange",
                    alpha=0.85, edgecolor="k", lw=1.2)
    plates = [(0, T_PLATE, "navy"), (T_PLATE, 2 * T_PLATE, "purple"),(2 * T_PLATE + H_GAP, 3 * T_PLATE + H_GAP, "purple"),(3 * T_PLATE + H_GAP, L_PIN, "navy"),]
    for z0, z1, color in plates:
        ax.add_patch(plt.Rectangle((z0 * 1000, -D_HOLE / 2 * 1000),(z1 - z0) * 1000, D_HOLE * 1000,  fill=False, edgecolor=color, linestyle=":", lw=1.0))
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_xlim(-0.4, L_PIN * 1000 + 0.4)
    ax.set_ylim(-D_HOLE * 1000 / 2 - 0.5, D_HOLE * 1000 / 2 + 0.5)
    ax.set_xlabel("axial position z (mm)")
    ax.set_ylabel("radial (mm)")
    ax.set_title(f"(b)  {pin.matkey}  —  d_end={pin.d_end*1000:.2f} mm, " f"d_mid={pin.d_mid*1000:.2f} mm\n"  f"navy = plate (support), purple = wing (load)")

    #subplot c bending moment diagram : m_y, m_x, |m|
    ax = fig.add_subplot(gs[1, 0])
    z = np.linspace(0, L_PIN, 600)
    M_y_arr = np.array([moments_at(zi, V_DESIGN, M_DRAG)[0] for zi in z])
    M_x_arr = np.array([moments_at(zi, V_DESIGN, M_DRAG)[1] for zi in z])
    M_combined_arr = np.hypot(M_y_arr, M_x_arr)
    ax.plot(z*1000, M_y_arr*1000, "steelblue", lw=1.4, label=r"$M_y$ (vertical, from V)")
    ax.plot(z*1000, M_x_arr*1000, "darkgreen", lw=1.4, label=r"$M_x$ (chordwise, from $M_{drag}$)")
    ax.plot(z*1000, M_combined_arr*1000, "firebrick", lw=2, label=r"$|M| = \sqrt{M_y^2 + M_x^2}$")
    for zs in (Z_TOP_PLATE, Z_TOP_WING, Z_BOT_WING, Z_BOT_PLATE): 
        ax.axvline(zs * 1000, ls=":", c="0.5", alpha=0.7)
    ax.set_xlabel("axial position z (mm)")
    ax.set_ylabel("M(z)  (mN·m)")
    ax.set_title("(c)  Bending moment along pin (4-pt bend + chordwise couple)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower center")

    #subplot d margin of safety : all four load cases
    ax = fig.add_subplot(gs[1, 1])
    s_aero = pin.stresses(V_DESIGN, M_DRAG)
    s_cruise = pin.stresses(V_CRUISE, M_DRAG * V_CRUISE / V_DESIGN)
    s_la = pin.stresses(V=V_LAUNCH, M_chord=M_LAUNCH)
    s_im = pin.stresses(V=V_IMPACT, M_chord=M_DRAG)
    sa_g = 0.5 * (s_aero["peak_vm"] - s_cruise["peak_vm"])
    sm_g = 0.5 * (s_aero["peak_vm"] + s_cruise["peak_vm"])
    gr = goodman(sa_g, sm_g, pin.mat["S_e"], pin.mat["S_ut"])
    ms_vals = [
        1.0 / (FS_FATIGUE * gr) - 1.0,
        pin.mat["S_y"] / FS_STATIC / s_aero["peak_vm"] - 1.0,
        pin.mat["S_y"] / FS_PROOF  / s_la["peak_vm"]   - 1.0,
        pin.mat["S_y"] / FS_PROOF  / s_im["peak_vm"]   - 1.0,
    ]
    labels_d = ["Cruise\n(Goodman)", "Aero 10g\n(FS=1.5)",
                 f"Launch {LAUNCH_G:.0f}g\n(FS_proof)",
                 f"Deploy impact\n(FS_proof)"]
    colors_d = ["blue", "orange", "red", "purple"]
    bars = ax.bar(labels_d, ms_vals, color=colors_d, alpha=0.82, edgecolor="k")
    ax.axhline(0.0, color="k", lw=1.4)
    ax.axhspan(min(min(ms_vals) * 1.3, -0.02), 0.0, color="red", alpha=0.08)
    ax.set_ylabel("Margin of safety  (allowable/actual − 1)")
    ax.set_title(f"(d) Margin of safety for all load cases\n-")
    ax.grid(axis="y", alpha=0.3)
    for b, ms in zip(bars, ms_vals):
        yoff = max(ms_vals) * 0.05 if ms >= 0 else min(ms_vals) * 0.12
        ax.text(b.get_x() + b.get_width() / 2.0, ms + yoff,
                f"{ms:+.2f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    plt.suptitle("Bio-inspired stepped pin (4-plate hinge): design summary", fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig("pin_design_summary.png", dpi=160, bbox_inches="tight")

# 3d rendering (pyvista)
def plot_3d_hinge(pin, b_plate=B_LUG_EFF, save="hinge_3d.png"):
    """render the 4:plate stack + stepped pin with pyvista."""
    import pyvista as pv
    pl = pv.Plotter(window_size=(1200, 900), off_screen=True)
    pl.set_background("white")

    plates = [
        (0, T_PLATE, "navy"), # plate top
        (T_PLATE, 2*T_PLATE, "purple"), # wing top
        (2*T_PLATE + H_GAP, 3*T_PLATE + H_GAP, "purple"), # wing bot
        (3*T_PLATE + H_GAP, L_PIN, "navy"), # plate bot
    ]
    for z0, z1, color in plates:
        plate = pv.Box(bounds=(-b_plate/2, b_plate/2, -b_plate/2, b_plate/2, z0, z1))
        pl.add_mesh(plate, color=color, opacity=0.30, show_edges=True, edge_color="black", line_width=1)
        for z in (z0, z1):
            ring = pv.Polygon(center=(0, 0, z), radius=D_HOLE/2, normal=(0, 0, 1), n_sides=64)
            pl.add_mesh(ring, color="white", style="wireframe", line_width=1.2)

    for z0, z1, d in [(0, Z_STEP_1, pin.d_end),  (Z_STEP_1, Z_STEP_2, pin.d_mid), (Z_STEP_2, L_PIN, pin.d_end)]:
        cyl = pv.Cylinder(center=(0, 0, (z0+z1)/2), direction=(0, 0, 1), radius=d/2, height=(z1-z0), resolution=72)
        pl.add_mesh(cyl, color="darkorange", specular=0.8, smooth_shading=True)

    arr = b_plate * 0.55
    for z in (Z_TOP_WING, Z_BOT_WING): # wing applies lift (red, +y)
        pl.add_mesh(pv.Arrow(start=(0, -arr, z), direction=(0, 1, 0), scale=arr*0.6, tip_length=0.3, shaft_radius=0.04), color="red")
    for z in (Z_TOP_PLATE, Z_BOT_PLATE): # plate reactions (blue, :y)
        pl.add_mesh(pv.Arrow(start=(0, arr, z), direction=(0, -1, 0),  scale=arr*0.6, tip_length=0.3, shaft_radius=0.04),  color="blue")

    pl.add_axes(line_width=3)
    pl.add_text(f"{pin.matkey}  d_end={pin.d_end*1000:.2f} mm  " f"d_mid={pin.d_mid*1000:.2f} mm", position="upper_edge", font_size=12, color="black")
    pl.camera_position = "iso"
    pl.show(screenshot=save)

# dynamic load check  (printed summary for any pin)
def dynamic_load_check(pin):
    """printing fancy plot [this took FOREVER to get right, pointless trial and error :(   ]"""
    s_aero = pin.stresses(V_DESIGN, M_DRAG)
    s_launch = pin.stresses(V=V_LAUNCH, M_chord=M_LAUNCH)
    s_impact = pin.stresses(V=V_IMPACT, M_chord=M_DRAG)

    rows = [
        ("Aero 10g", V_DESIGN, FS_STATIC, s_aero),
        (f"Launch {LAUNCH_G:.0f}g", V_LAUNCH, FS_PROOF, s_launch),
        ("Deploy impact", V_IMPACT, FS_PROOF, s_impact),
    ]
    print(f"\n{'='*60}")
    print(f"DYNAMIC LOAD CHECK:  {pin.matkey}  "
          f"d_end={pin.d_end*1000:.2f} mm  d_mid={pin.d_mid*1000:.2f} mm")
    print(f"Deploy impact: F_stop={F_STOP:.0f} N  "
          f"({F_STOP/V_DESIGN:.1f}× aero)  ω_stop={np.degrees(_omega):.0f}°/s")
    print(f"{'='*60}")
    print(f"  {'Case':<18} {'V (N)':>8}  {'sigma_VM (MPa)':>11}  {'Allowable':>10}  {'MS':>7}")
    print(f"  {'-'*62}")
    for label, V, fs, s in rows:
        allowable = pin.mat["S_y"] / fs / 1e6
        ms = pin.mat["S_y"] / fs / s["peak_vm"] - 1.0
        flag = "works" if ms >= 0 else "fails"
        print(f"{label:<18} {V:>8.1f}  {s['peak_vm']/1e6:>11.1f}" f"{allowable:>10.1f}  {ms:>+7.3f}  {flag}")
    print()

# __main__
if __name__ == "__main__":
    pin = run_optimizer()
    if pin is not None:
        dynamic_load_check(pin)
        plot_3d_hinge(pin)
    with open("optimized_pin.json", "r") as f:
        out_data = json.load(f)
    pin = SteppedPin(out_data["material"], out_data["d_end"], out_data["d_mid"])
    print("Predicted Hinge Life:")
    print(pin.predict_life())