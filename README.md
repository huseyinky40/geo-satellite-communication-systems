# 🛰️ GEO Satellite Communication Systems

A Python simulation suite that applies communication theory to a real Geostationary Earth Orbit (GEO) satellite link, modelled after the **Türksat 4A** (Ku-band, 42°E) reference mission.

The central question: *"How does a signal survive 200+ dB of free-space path loss?"*

---

## 📡 Overview

GEO satellites orbit at **35,786 km** altitude, providing wide-area coverage to over a third of the Earth's surface. This project analytically and numerically characterises the end-to-end link using six core equations — free-space path loss, the link budget, the AWGN channel model, BER for multiple modulation schemes, Shannon capacity, and ITU-R P.838 rain attenuation.

---

## 📊 Simulation Figures

| Figure | Description |
|--------|-------------|
| **Fig 1** | GEO satellite system block diagram (Uplink → Transponder → Downlink) |
| **Fig 2** | BER vs E_b/N₀ — BPSK, QPSK, 8-PSK, 16-QAM (theory + Monte Carlo validation) |
| **Fig 3** | Ku-band link budget waterfall — step-by-step C/N = 18.64 dB derivation |
| **Fig 4** | Free-space path loss vs distance for C, Ku, and Ka-band |
| **Fig 5** | Shannon channel capacity and spectral efficiency vs SNR |
| **Fig 6** | Rain attenuation vs rainfall rate — ITU-R P.838 model (C / Ku / Ka-band) |

---

## 🔑 Key Results

| Analysis | Result |
|----------|--------|
| Ku-band FSPL (12 GHz, 35,786 km) | **205.10 dB** |
| Link Carrier-to-Noise Ratio (C/N) | **18.64 dB** |
| QPSK BER = 10⁻⁶ threshold | E_b/N₀ = **10.5 dB** |
| Shannon Capacity (36 MHz transponder) | **223.6 Mbps** |
| Ka-band rain attenuation (42 mm/h, 5 km) | **≈ 37 dB** |
| Monte Carlo vs. Analytical BER | **±5% agreement** |

---

## 📁 Repository Structure

```
geo-satellite-communication-systems/
│
├── geo_satellite_simulation.py   # Main simulation script
├── figures/                      # Simulation output figures
│   ├── fig1_block_diagram.png
│   ├── fig2_ber_vs_snr.png
│   ├── fig3_link_budget.png
│   ├── fig4_path_loss.png
│   ├── fig5_shannon_capacity.png
│   └── fig6_rain_attenuation.png
├── report.pdf                    # Full project report
├── presentation.pdf              # Presentation slides
└── README.md
```

---

## ⚙️ Requirements

**Python 3.8+**

```bash
pip install numpy scipy matplotlib
```

---

## 🚀 Usage

```bash
python geo_satellite_simulation.py
```

All six figures are saved to the `figures/` directory. Key numerical results are printed to stdout:

```
GEO Satellite Communication Systems — Simulation
====================================================
  Ku-band FSPL (12 GHz, 35 786 km) : 205.10 dB
  Carrier-to-Noise ratio (C/N)     : 18.64 dB
  Shannon capacity (B=36 MHz)      : 223.6 Mbps
  ...
```

---

## 📐 Theory

| Equation | Formula | Purpose |
|----------|---------|---------|
| Free-Space Path Loss | `FSPL = 20·log₁₀(4πdf/c)` | Dominant link impairment |
| Link Budget | `C/N = EIRP + G/T − FSPL − 10·log₁₀(kB) − L` | End-to-end link quality |
| AWGN Channel | `r(t) = s(t) + n(t), n ~ N(0, σ²)` | Channel model |
| BER (BPSK/QPSK) | `P_e = Q(√(2·E_b/N₀))` | Modulation performance |
| Shannon Capacity | `C = B · log₂(1 + SNR)` | Theoretical throughput limit |
| Rain Attenuation | `γ_r = k · R^α → A = γ_r · d_eff` | ITU-R P.838 impairment model |

---

## 📡 Reference System — Türksat 4A (Ku-Band)

| Parameter | Value | Unit |
|-----------|-------|------|
| GEO Altitude | 35,786 | km |
| Downlink Frequency | 12 | GHz |
| Satellite EIRP | 52 | dBW |
| Earth Station G/T | 20 | dB/K |
| Transponder Bandwidth | 36 | MHz |
| System Noise Temperature | 150 | K |

---

## 🔭 Modulation Comparison

| Scheme | Bits/Symbol | E_b/N₀ for BER=10⁻⁶ | Spectral Efficiency |
|--------|-------------|----------------------|---------------------|
| BPSK | 1 | 10.5 dB | ~1 bps/Hz |
| **QPSK** | **2** | **10.5 dB** | **~2 bps/Hz** ✅ DVB-S2 baseline |
| 8-PSK | 3 | 14.0 dB | ~3 bps/Hz |
| 16-QAM | 4 | 17.5 dB | ~4 bps/Hz |

> QPSK achieves identical BER to BPSK at **2× the spectral efficiency** — the reason it is the DVB-S2 baseline modulation.

---

## 🌧️ Rain Attenuation at Istanbul (R = 42 mm/h, Zone K)

| Band | Frequency | Attenuation | Impact |
|------|-----------|-------------|--------|
| C-band | 6 GHz | ≈ 1.0 dB | Negligible ✅ |
| Ku-band | 14 GHz | ≈ 7.5 dB | Manageable ⚠️ |
| Ka-band | 30 GHz | ≈ 37 dB | Catastrophic ❌ |

---

## 📚 References

1. Proakis & Salehi, *Digital Communications*, 5th ed., McGraw-Hill, 2008
2. ETSI EN 302 307 — DVB-S2 Standard
3. ITU-R P.838-3 — Specific Attenuation Model for Rain
4. ITU-R P.618-13 — Earth-Space Propagation Data
5. Pratt, Allnutt & Bostian, *Satellite Communications*, 2nd ed., Wiley, 2003

---

> *Developed as part of a communication theory course project at Istanbul Arel University (2026).*
