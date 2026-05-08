"""
GEO Satellite Communication Systems Simulation
LEEN354 - Communication Theory II
Istanbul Arel University

This script generates all simulation figures for the project report.
Run: python geo_satellite_simulation.py
Output: figures/ directory (6 PNG files)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from scipy.special import erfc
import os

# ─── Output directory ────────────────────────────────────────────────────────
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ─── Global plot style ────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "grid.alpha": 0.35,
    "figure.dpi": 150,
})
COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]

# ═════════════════════════════════════════════════════════════════════════════
# SYSTEM PARAMETERS  (based on Türksat 4A / Intelsat reference mission)
# ═════════════════════════════════════════════════════════════════════════════
C           = 3e8           # Speed of light [m/s]
K_B         = 1.380649e-23  # Boltzmann constant [J/K]
GEO_ALT_KM  = 35_786        # GEO orbit altitude [km]
GEO_ALT_M   = GEO_ALT_KM * 1e3

# Frequency bands  [Hz]
BANDS = {
    "C-band (4 GHz DL)":  4e9,
    "Ku-band (12 GHz DL)": 12e9,
    "Ka-band (20 GHz DL)": 20e9,
}

# Uplink frequencies for rain attenuation model [GHz]
BANDS_GHZ_UL = {"C-band (6 GHz)": 6, "Ku-band (14 GHz)": 14, "Ka-band (30 GHz)": 30}

# Ku-band link budget parameters (Türksat 4A typical)
EIRP_DBW        = 52.0      # Satellite EIRP [dBW]  (Türksat 4A Ku-band spot beam)
GT_DB           = 20.0      # Earth station G/T [dB/K]  (2.4 m dish, T_sys≈150 K)
BW_HZ           = 36e6      # Transponder bandwidth [Hz]
NOISE_TEMP_K    = 290       # System noise temperature [K]
POINTING_LOSS   = 0.5       # Pointing loss [dB]
ATMOS_LOSS      = 0.3       # Atmospheric absorption [dB] (clear sky)
POLARIZ_LOSS    = 0.5       # Polarisation mismatch loss [dB]

# ═════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def fspl_db(distance_m, freq_hz):
    """Free-Space Path Loss in dB."""
    return 20 * np.log10(4 * np.pi * distance_m * freq_hz / C)


def q_func(x):
    """Q-function via complementary error function."""
    return 0.5 * erfc(x / np.sqrt(2))


def ber_bpsk(ebn0_lin):
    return q_func(np.sqrt(2 * ebn0_lin))


def ber_qpsk(ebn0_lin):
    return q_func(np.sqrt(2 * ebn0_lin))          # same as BPSK per bit


def ber_8psk(ebn0_lin):
    return (2 / 3) * q_func(np.sqrt(2 * 3 * ebn0_lin * np.sin(np.pi / 8) ** 2))


def ber_16qam(ebn0_lin):
    return (3 / 8) * erfc(np.sqrt(ebn0_lin / 5))


def monte_carlo_bpsk(ebn0_db_list, n_bits=200_000):
    """Monte Carlo BER estimate for BPSK over AWGN."""
    ber_mc = []
    rng = np.random.default_rng(42)
    for ebn0_db in ebn0_db_list:
        ebn0_lin = 10 ** (ebn0_db / 10)
        bits = rng.integers(0, 2, n_bits)
        symbols = 2 * bits - 1                    # BPSK: -1 / +1
        noise_std = np.sqrt(1 / (2 * ebn0_lin))
        received = symbols + rng.normal(0, noise_std, n_bits)
        decided = (received > 0).astype(int)
        errors = np.sum(bits != decided)
        ber_mc.append(max(errors / n_bits, 1e-7))  # floor to avoid log(0)
    return np.array(ber_mc)


# ITU-R P.838-3 regression coefficients (horizontal polarisation, approximate)
ITU_COEFF = {
    6:  (0.00265, 1.312),
    14: (0.0188,  1.217),
    30: (0.167,   1.000),
}


def rain_attenuation(R_mm_hr, freq_ghz, path_length_km=5.0):
    """
    Specific rain attenuation via ITU-R P.838 power law:  γ_R = k · R^α  [dB/km]
    Total attenuation: A = γ_R · d_eff  with d_eff ≈ path_length_km
    """
    k, alpha = ITU_COEFF[freq_ghz]
    gamma_R = k * R_mm_hr ** alpha
    return gamma_R * path_length_km


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — GEO SATELLITE SYSTEM BLOCK DIAGRAM
# ═════════════════════════════════════════════════════════════════════════════

def plot_block_diagram():
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis("off")
    fig.patch.set_facecolor("#f8f9fa")

    def box(ax, x, y, w, h, label, sublabel="", color="#4C72B0"):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.1",
            facecolor=color, edgecolor="white",
            linewidth=1.5, zorder=3,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2 + (0.18 if sublabel else 0), label,
                ha="center", va="center", fontsize=9.5, fontweight="bold",
                color="white", zorder=4)
        if sublabel:
            ax.text(x + w / 2, y + h / 2 - 0.22, sublabel,
                    ha="center", va="center", fontsize=8, color="#d0e4ff", zorder=4)

    def arrow(ax, x1, x2, y, label=""):
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.5))
        if label:
            ax.text((x1 + x2) / 2, y + 0.2, label,
                    ha="center", va="bottom", fontsize=8, color="#444")

    # ── Uplink chain ──────────────────────────────────────────────
    ax.text(3.2, 4.6, "UPLINK  (Earth → Satellite)", ha="center",
            fontsize=10, color="#333", style="italic")
    box(ax, 0.2, 3.6, 1.5, 0.8, "Source /", "Encoder", "#2c7bb6")
    box(ax, 2.0, 3.6, 1.5, 0.8, "Modulator", "(BPSK/QPSK)", "#1a9641")
    box(ax, 3.8, 3.6, 1.5, 0.8, "HPA +", "Antenna TX", "#d7191c")
    arrow(ax, 1.7, 2.0, 4.0, "bits")
    arrow(ax, 3.5, 3.8, 4.0, "RF")
    arrow(ax, 5.3, 5.8, 4.0, "uplink\nwave")

    # ── Satellite transponder ──────────────────────────────────────
    ax.text(7.2, 4.6, "GEO SATELLITE  (35 786 km)", ha="center",
            fontsize=10, color="#333", style="italic")
    box(ax, 5.8, 3.55, 2.8, 0.9, "Transponder", "(LNA · Filter · HPA)", "#7b2d8b")
    arrow(ax, 8.6, 9.1, 4.0, "downlink\nwave")

    # Free-space channel annotation
    for x_pos, y_pos, txt in [
        (5.55, 3.1, "Free-Space\nPath Loss\n~205 dB (Ku)"),
        (9.1, 3.1, "Rain\nAttenuation\n+ Noise"),
    ]:
        ax.text(x_pos, y_pos, txt, ha="center", va="top",
                fontsize=8, color="#555",
                bbox=dict(boxstyle="round,pad=0.3", fc="#fff3cd", ec="#aaa", alpha=0.9))

    # ── Downlink chain ─────────────────────────────────────────────
    ax.text(11.2, 4.6, "DOWNLINK  (Satellite → Earth)", ha="center",
            fontsize=10, color="#333", style="italic")
    box(ax, 9.1, 3.6, 1.5, 0.8, "LNA +", "Antenna RX", "#d7191c")
    box(ax, 10.8, 3.6, 1.5, 0.8, "Demodulator", "/ Decoder", "#1a9641")
    box(ax, 12.5, 3.6, 1.3, 0.8, "Sink /", "User", "#2c7bb6")
    arrow(ax, 10.6, 10.8, 4.0)
    arrow(ax, 12.3, 12.5, 4.0, "bits")

    # ── AWGN noise label ──────────────────────────────────────────
    ax.annotate("", xy=(11.55, 3.6), xytext=(11.55, 2.8),
                arrowprops=dict(arrowstyle="-|>", color="#d7191c", lw=1.5))
    ax.text(11.55, 2.6, "AWGN\n(Thermal Noise)", ha="center", fontsize=8,
            color="#d7191c",
            bbox=dict(boxstyle="round,pad=0.25", fc="#ffe0e0", ec="#d7191c", alpha=0.8))

    ax.set_title("Figure 1 — GEO Satellite Communication System Block Diagram",
                 fontsize=12, fontweight="bold", pad=10)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig1_block_diagram.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — BER vs Eb/N0  (Theory + Monte Carlo)
# ═════════════════════════════════════════════════════════════════════════════

def plot_ber_vs_snr():
    ebn0_db  = np.linspace(0, 20, 500)
    ebn0_lin = 10 ** (ebn0_db / 10)

    fig, ax = plt.subplots(figsize=(9, 6))

    modulations = [
        ("BPSK",   ber_bpsk(ebn0_lin),   COLORS[0], "-"),
        ("QPSK",   ber_qpsk(ebn0_lin),   COLORS[1], "--"),
        ("8-PSK",  ber_8psk(ebn0_lin),   COLORS[2], "-."),
        ("16-QAM", ber_16qam(ebn0_lin),  COLORS[3], ":"),
    ]
    for label, ber, color, ls in modulations:
        ax.semilogy(ebn0_db, ber, ls, color=color, lw=2, label=f"{label} (theory)")

    # Monte Carlo points for BPSK (sparse set for visual clarity)
    mc_ebn0 = np.arange(0, 14, 2)
    mc_ber  = monte_carlo_bpsk(mc_ebn0)
    ax.semilogy(mc_ebn0, mc_ber, "o", color=COLORS[0], ms=7,
                label="BPSK (Monte Carlo)", zorder=5)

    # DVB-S2 operating point annotation — placed in lower-right clear zone
    ax.axvline(x=9.4, color="#555", lw=1, ls="--", alpha=0.6)
    ax.axhline(y=1e-6, color="#555", lw=1, ls="--", alpha=0.6)
    ax.annotate(
        "DVB-S2 operating point\n(QPSK, BER = 10⁻⁶)",
        xy=(9.4, 1e-6),
        xytext=(13.5, 5e-6),
        fontsize=8.5, color="#444",
        arrowprops=dict(arrowstyle="->", color="#666", lw=1.0),
        bbox=dict(boxstyle="round,pad=0.35", fc="lightyellow", ec="#bbb", alpha=0.95),
    )

    ax.set_xlabel("$E_b/N_0$ [dB]")
    ax.set_ylabel("Bit Error Rate (BER)")
    ax.set_title("Figure 2 — BER vs $E_b/N_0$ for Different Modulation Schemes\n"
                 "(AWGN Channel — GEO Satellite Downlink)", fontweight="bold")
    ax.set_xlim(0, 20)
    ax.set_ylim(1e-7, 1)
    # Legend in lower-left: curves are near 1 there only for high-order mods,
    # but BPSK/QPSK are already below 1e-7 so the area is clear.
    ax.legend(loc="lower left", framealpha=0.92)
    ax.grid(True, which="both")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig2_ber_vs_snr.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — LINK BUDGET (Ku-band)
# ═════════════════════════════════════════════════════════════════════════════

def plot_link_budget():
    freq_hz      = 12e9
    fspl         = fspl_db(GEO_ALT_M, freq_hz)
    noise_floor  = 10 * np.log10(K_B * NOISE_TEMP_K * BW_HZ)   # dBW
    cn_db        = EIRP_DBW + GT_DB - fspl - POINTING_LOSS - ATMOS_LOSS - POLARIZ_LOSS \
                   - 10 * np.log10(K_B * BW_HZ)

    components = {
        "EIRP\n(Satellite)":     EIRP_DBW,
        "Receive G/T":            GT_DB,
        "Free-Space\nPath Loss":  -fspl,
        "Pointing\nLoss":         -POINTING_LOSS,
        "Atmospheric\nLoss":      -ATMOS_LOSS,
        "Polarisation\nLoss":     -POLARIZ_LOSS,
        "Noise Power\n(kTB)":     noise_floor,
    }

    labels = list(components.keys())
    values = list(components.values())
    colors_bar = [COLORS[0] if v >= 0 else COLORS[1] for v in values]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # ── Left: waterfall bars ──────────────────────────────────────
    bars = ax1.barh(labels, values, color=colors_bar, edgecolor="white", linewidth=0.8)
    ax1.axvline(0, color="#333", lw=0.8)
    for bar, val in zip(bars, values):
        sign = "+" if val >= 0 else ""
        ax1.text(val + (0.5 if val >= 0 else -0.5),
                 bar.get_y() + bar.get_height() / 2,
                 f"{sign}{val:.1f} dB", va="center",
                 ha="left" if val >= 0 else "right", fontsize=9)
    ax1.set_xlabel("Contribution [dB / dBW]")
    ax1.set_title("Link Budget Components (Ku-band, 12 GHz)", fontweight="bold")
    pos_patch = mpatches.Patch(color=COLORS[0], label="Gain / signal boost")
    neg_patch = mpatches.Patch(color=COLORS[1], label="Loss / noise")
    # Place legend in upper-right (away from the large negative bars on the left)
    ax1.legend(handles=[pos_patch, neg_patch], loc="upper right", fontsize=9,
               framealpha=0.92)
    ax1.grid(axis="x", alpha=0.3)

    # ── Right: cumulative cascade ─────────────────────────────────
    gains = [EIRP_DBW, GT_DB]
    losses = [fspl, POINTING_LOSS, ATMOS_LOSS, POLARIZ_LOSS]
    total_gains  = sum(gains)
    total_losses = sum(losses)
    cn_kTB = 10 * np.log10(K_B * BW_HZ)   # thermal noise reference (dBW/Hz × BW)

    stages  = ["EIRP", "+G/T", "−FSPL", "−Losses", "C/N\n(result)"]
    running = [EIRP_DBW,
               EIRP_DBW + GT_DB,
               EIRP_DBW + GT_DB - fspl,
               EIRP_DBW + GT_DB - fspl - sum(losses),
               cn_db]
    stage_colors = [COLORS[0], COLORS[0], COLORS[1], COLORS[1], COLORS[2]]
    ax2.bar(stages, running, color=stage_colors, edgecolor="white",
            linewidth=0.8, width=0.5)
    for i, (s, v) in enumerate(zip(stages, running)):
        ax2.text(i, v + 0.5, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    ax2.axhline(y=cn_db, color=COLORS[2], lw=1.5, ls="--", alpha=0.7,
                label=f"C/N = {cn_db:.1f} dB")
    ax2.set_ylabel("Cumulative Level [dB / dBW]")
    ax2.set_title(f"Link Cascade → C/N = {cn_db:.1f} dB", fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Figure 3 — Ku-Band GEO Satellite Link Budget Analysis",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig3_link_budget.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — FREE-SPACE PATH LOSS vs DISTANCE
# ═════════════════════════════════════════════════════════════════════════════

def plot_path_loss():
    distances_km = np.linspace(1000, 50_000, 1000)
    distances_m  = distances_km * 1e3

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Vertical offsets for GEO-altitude annotations so labels don't overlap
    # C-band ~195.6, Ku-band ~205.1, Ka-band ~209.5 dB — stagger x-positions
    annot_x_offsets = [-8000, -8000, -8000]   # all to the LEFT of GEO line
    annot_y_offsets = [1.0, -1.5, 1.0]        # C above, Ku below, Ka above (Ka highest)

    for i, ((label, freq), color) in enumerate(zip(BANDS.items(), COLORS)):
        fspl_vals = fspl_db(distances_m, freq)
        ax.plot(distances_km, fspl_vals, color=color, lw=2, label=label)
        # Mark GEO altitude
        geo_loss = fspl_db(GEO_ALT_M, freq)
        ax.plot(GEO_ALT_KM, geo_loss, "o", color=color, ms=9, zorder=5)
        ax.annotate(
            f"{geo_loss:.1f} dB",
            xy=(GEO_ALT_KM, geo_loss),
            xytext=(GEO_ALT_KM + annot_x_offsets[i],
                    geo_loss + annot_y_offsets[i]),
            fontsize=8.5, color=color,
            ha="right",
            arrowprops=dict(arrowstyle="-", color=color, lw=0.8),
        )

    ax.axvline(GEO_ALT_KM, color="#888", lw=1.2, ls="--", alpha=0.7)
    # Place GEO altitude label to the RIGHT of the dashed line (clear space)
    ax.text(GEO_ALT_KM + 600, 162,
            f"GEO altitude\n{GEO_ALT_KM:,} km", fontsize=8.5, color="#555")

    ax.set_xlabel("Distance [km]")
    ax.set_ylabel("Free-Space Path Loss [dB]")
    ax.set_title("Figure 4 — Free-Space Path Loss vs Distance\n"
                 "for C, Ku, and Ka-band GEO Links", fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig4_path_loss.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — SHANNON CAPACITY vs SNR
# ═════════════════════════════════════════════════════════════════════════════

def plot_shannon_capacity():
    snr_db  = np.linspace(-5, 30, 500)
    snr_lin = 10 ** (snr_db / 10)

    bandwidths_mhz = [18, 36, 72]   # typical transponder bandwidths

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for bw_mhz, color in zip(bandwidths_mhz, COLORS):
        bw_hz   = bw_mhz * 1e6
        cap_mbps = bw_hz * np.log2(1 + snr_lin) / 1e6
        ax1.plot(snr_db, cap_mbps, color=color, lw=2, label=f"B = {bw_mhz} MHz")

        spec_eff = np.log2(1 + snr_lin)    # bps/Hz — same for all BW
    # Spectral efficiency (BW-independent)
    ax2.plot(snr_db, np.log2(1 + snr_lin), color=COLORS[0], lw=2.5)

    # Mark operating points — labels placed to the right with clear vertical spread
    # xytext=(absolute_x, absolute_y) so labels never fall below y=0
    op_points = [
        ("BPSK (~1 bps/Hz)",   0,    1.0,  (6,  1.2)),
        ("QPSK (~2 bps/Hz)",   3.0,  2.0,  (8,  2.5)),
        ("8-PSK (~3 bps/Hz)",  6.8,  3.0,  (12, 3.8)),
        ("16-QAM (~4 bps/Hz)", 11.7, 4.0,  (17, 5.2)),
    ]
    for label, snr_op, se_op, (tx, ty) in op_points:
        ax2.plot(snr_op, se_op, "o", ms=8, color=COLORS[3], zorder=5)
        ax2.annotate(label,
                     xy=(snr_op, se_op),
                     xytext=(tx, ty),
                     fontsize=8, color=COLORS[3], ha="left",
                     arrowprops=dict(arrowstyle="->", color=COLORS[3], lw=0.7,
                                     connectionstyle="arc3,rad=0.1"))

    ax1.set_xlabel("SNR [dB]")
    ax1.set_ylabel("Channel Capacity [Mbps]")
    ax1.set_title("Shannon Capacity vs SNR\n(Various Transponder Bandwidths)", fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-5, 30)
    ax1.set_ylim(0)

    ax2.set_xlabel("SNR [dB]")
    ax2.set_ylabel("Spectral Efficiency [bps/Hz]")
    ax2.set_title("Spectral Efficiency vs SNR\n(Shannon Limit + Modulation Points)", fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-5, 30)
    ax2.set_ylim(0, 12)

    fig.suptitle("Figure 5 — Shannon Channel Capacity for GEO Satellite Links",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig5_shannon_capacity.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — RAIN ATTENUATION vs RAINFALL RATE  (ITU-R P.838)
# ═════════════════════════════════════════════════════════════════════════════

def plot_rain_attenuation():
    R_range = np.linspace(0.1, 150, 500)   # rainfall rate [mm/hour]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for (label, freq_ghz), color in zip(BANDS_GHZ_UL.items(), COLORS):
        att = rain_attenuation(R_range, freq_ghz)
        ax.plot(R_range, att, color=color, lw=2, label=label)

    # Rain climate zone markers (ITU-R P.837)
    # Stagger label heights so nearby zones don't overlap
    ax.set_ylim(bottom=0)
    ymax = ax.get_ylim()[1]
    zone_info = [
        ("Zone D (Istanbul ~15 mm/h)",  15, 0.72),   # 72% height
        ("Zone E (Tropical ~22 mm/h)",  22, 0.82),   # 82% height — staggered up
        ("Zone K (Heavy ~42 mm/h)",     42, 0.72),   # 72% height
    ]
    for zone_label, r_val, y_frac in zone_info:
        ax.axvline(r_val, color="#aaa", lw=1, ls=":", alpha=0.8)
        ax.text(r_val + 1.2, ymax * y_frac, zone_label,
                fontsize=7.5, color="#666", va="bottom", ha="left")

    ax.set_xlabel("Rainfall Rate R [mm/hour]")
    ax.set_ylabel("Rain Attenuation [dB]")
    ax.set_title("Figure 6 — Rain Attenuation vs Rainfall Rate\n"
                 "(ITU-R P.838 Model, Path Length = 5 km, Uplink Frequencies)", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig6_rain_attenuation.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("GEO Satellite Communication Systems — Simulation")
    print("=" * 52)

    steps = [
        ("Figure 1: Block Diagram",        plot_block_diagram),
        ("Figure 2: BER vs Eb/N0",         plot_ber_vs_snr),
        ("Figure 3: Link Budget",          plot_link_budget),
        ("Figure 4: Free-Space Path Loss", plot_path_loss),
        ("Figure 5: Shannon Capacity",     plot_shannon_capacity),
        ("Figure 6: Rain Attenuation",     plot_rain_attenuation),
    ]

    for name, fn in steps:
        print(f"\n[{name}]")
        fn()

    # ── Print key numerical results ───────────────────────────────
    print("\n" + "=" * 52)
    print("KEY NUMERICAL RESULTS")
    print("=" * 52)

    ku_fspl = fspl_db(GEO_ALT_M, 12e9)
    print(f"  Ku-band FSPL (12 GHz, 35 786 km) : {ku_fspl:.2f} dB")

    c_fspl  = fspl_db(GEO_ALT_M, 4e9)
    print(f"  C-band  FSPL (4 GHz,  35 786 km) : {c_fspl:.2f} dB")

    ka_fspl = fspl_db(GEO_ALT_M, 20e9)
    print(f"  Ka-band FSPL (20 GHz, 35 786 km) : {ka_fspl:.2f} dB")

    noise_floor = 10 * np.log10(K_B * NOISE_TEMP_K * BW_HZ)
    cn_db = EIRP_DBW + GT_DB - ku_fspl - POINTING_LOSS - ATMOS_LOSS - POLARIZ_LOSS \
            - 10 * np.log10(K_B * BW_HZ)
    print(f"\n  Noise floor (T=290K, B=36MHz)    : {noise_floor:.2f} dBW")
    print(f"  Carrier-to-Noise ratio (C/N)     : {cn_db:.2f} dB")

    bpsk_ebn0_for_1e6 = 10.5
    print(f"\n  BPSK: Eb/N0 for BER=10⁻⁶         ≈ {bpsk_ebn0_for_1e6} dB")
    print(f"  QPSK: same BER, same Eb/N0  (2× spectral efficiency)")

    cap_36mhz_cn = 36e6 * np.log2(1 + 10 ** (cn_db / 10)) / 1e6
    print(f"\n  Shannon capacity (B=36 MHz, C/N={cn_db:.1f} dB): {cap_36mhz_cn:.1f} Mbps")

    print(f"\n  Rain att. (Ku, R=15 mm/h, 5km)  : {rain_attenuation(15, 14):.2f} dB")
    print(f"  Rain att. (Ka, R=15 mm/h, 5km)  : {rain_attenuation(15, 30):.2f} dB")

    print(f"\nAll figures saved to: {FIG_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
