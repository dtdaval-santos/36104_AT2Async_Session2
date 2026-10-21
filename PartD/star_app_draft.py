# AppStar (draft).  Run with:  streamlit run star_app_draft.py
#
# Python runs once: it precomputes the star's state over a grid of
# masses and ages, then ships one Altair chart whose sliders are
# Vega parameters. Dragging them filters the grid in the browser,
# so the star updates live, with no Python rerun.
#
# Draft changes from star_app.py (Metallicity section of Part D):
#   1. metallicity is a log10 st.slider; Z and zr follow from it, and
#      releasing it reruns the script and rebuilds the grid
#   2. pair-instability branch: 140 to 260 suns with Z < 0.001 leaves
#      no remnant
#   3. the right-hand panel is an HR diagram built in Altair (log-log,
#      temperature reversed) instead of the mass-age phase plane
# Mass and age stay live in the browser.

import math

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

SUN_T = 5772.0

st.set_page_config(page_title="AppStar", layout="wide")

st.markdown("""<style>
.stApp {background-color: #000000;}
.stApp, .stApp p, .stApp label {color: #e8e8e8;}
.block-container {padding-top: 0.4rem; padding-bottom: 0.3rem;
                  max-width: 1000px;}
header[data-testid="stHeader"] {display: none;}
h1, h2, h3 {padding-top: 0 !important; margin: 0 0 0.2rem !important;
            color: #f0f0f0;}
form.vega-bindings {display: flex; justify-content: center;
                    gap: 3rem; margin-top: 0.6rem;
                    color: #e8e8e8; font-weight: 700;}
form.vega-bindings input[type="range"] {width: 240px;}
</style>""", unsafe_allow_html=True)

st.markdown("### AppStar")

alt.data_transformers.disable_max_rows()

lz = round(st.slider("log10 metallicity", -4.0, -1.4, -1.7,
                     step=0.05), 2)
Z = 10 ** lz      # metallicity; the Sun is about 0.02
zr = Z / 0.02


def bb_rgb(T):
    # Approximate black-body colour, valid from about 1000 to 40000 K.
    t = T / 100.0
    r = 255.0 if t <= 66 else 329.7 * (t - 60) ** -0.1332
    g = 99.47 * math.log(t) - 161.1 if t <= 66 else 288.1 * (t - 60) ** -0.0755
    if t >= 66:
        b = 255.0
    elif t <= 19:
        b = 0.0
    else:
        b = 138.5 * math.log(t - 10) - 305.0
    return tuple(min(255.0, max(0.0, v)) / 255 for v in (r, g, b))


def rgb_str(T):
    r, g, b = (int(round(255 * c)) for c in bb_rgb(min(T, 40000)))
    return f"rgb({r},{g},{b})"


def star_state(mass, age):
    # the same rules as the course page
    L = mass ** 3.5 * zr ** -0.1        # luminosity, suns  [uses zr]
    R = mass ** 0.8                     # radius, suns
    T = SUN_T * (L / R ** 2) ** 0.25    # surface temperature, K
    t_pre = 0.03 * mass ** -1.5
    t_ms = (10.0 * mass ** -2.5 * (1 + 2.5 * math.exp(-mass / 0.12))
            + 0.0025)
    t_g = 1.15 * t_ms

    if age <= t_pre:
        phase = "protostar"
    elif age <= t_ms:
        phase = "main sequence"
    elif mass < 0.25:
        phase = "white dwarf"           # fully convective: no giant
    elif mass < 8:
        phase = "red giant" if age <= t_g else "white dwarf"
    elif age <= t_g:
        frac = (age - t_ms) / (t_g - t_ms)
        phase = "blue supergiant" if frac < 0.4 else "red supergiant"
    elif age <= 1.10 * t_g:
        phase = "supernova"
    elif Z < 0.001 and 140 <= mass <= 260:
        phase = "no remnant"            # pair instability: blown apart
    elif mass < 18 + 7 * zr:            # remnant boundary  [uses zr]
        phase = "neutron star"
    else:
        phase = "black hole"

    T_show, L_show, R_show = float(T), L, R
    if phase == "protostar":
        T_show, L_show, R_show = 0.75 * T, 2 * L, 3 * R
    elif phase == "red giant":
        T_show = 3900.0
        R_show = max(R * 60, 10.0)
        L_show = R_show ** 2 * (T_show / SUN_T) ** 4
    elif phase == "blue supergiant":
        frac = (age - t_ms) / (t_g - t_ms)
        T_show = 12000.0
        R_show = 30 + 120 * frac
        L_show = R_show ** 2 * (T_show / SUN_T) ** 4
    elif phase == "red supergiant":
        frac = (age - t_ms) / (t_g - t_ms)
        T_show = 3500.0
        R_show = min(200 + 900 * frac, 900)
        L_show = R_show ** 2 * (T_show / SUN_T) ** 4
    elif phase == "white dwarf":
        cool = max(age - (t_ms if mass < 0.25 else t_g), 0.001)
        T_show = float(np.clip(60000.0 * (0.01 / cool) ** 0.3,
                               3500, 150000))
        R_show = 0.009
        L_show = R_show ** 2 * (T_show / SUN_T) ** 4
    elif phase == "supernova":
        # no luminosity: an explosion is an event, not an equilibrium state,
        # and its ~5e9 suns would stretch an HR luminosity axis by four
        # decades to hold one transient point
        T_show, L_show, R_show = 8000.0, None, None
    elif phase == "no remnant":
        T_show, L_show, R_show = None, None, None
    elif phase == "neutron star":
        T_show, L_show, R_show = 1e6, None, 1.7e-5
    elif phase == "black hole":
        T_show, L_show, R_show = None, None, 4.2e-6 * mass / 10
    return phase, T_show, L_show, R_show, t_ms


# ---- the grid: one row per slider combination ------------------------
lms = [round(-1.0 + 0.05 * k, 2) for k in range(70)]   # mass 0.1 to 282
las = [round(-4.0 + 0.06 * k, 2) for k in range(127)]  # age 1e-4 to 3631

rows = []
for lm in lms:
    for la in las:
        m = 10.0 ** lm
        a = 10.0 ** la
        phase, T, L, R, t_ms = star_state(m, a)
        if phase == "no remnant":
            colour, px = "rgb(0,0,0)", 1.0     # nothing left to draw
        elif phase == "black hole":
            colour, px = "rgb(16,16,16)", 40.0
        elif phase == "neutron star":
            colour, px = "#CDE7FF", 6.0
        elif phase == "supernova":
            colour, px = "#FFD27D", 150.0
        else:
            colour = rgb_str(T)
            px = float(np.clip(14 + 26 * (np.log10(R) + 2.2), 5, 150))
        rows.append(dict(
            lm=lm, la=la, mass=m, age=a,
            temp_K=T, lum=L, rad=R,
            colour=colour, size=px ** 2, phase=phase,
            massage=f"mass {m:.2g} suns, age {a:.2g} Gyr",
            temp=f"surface {T:,.0f} K" if T else "",
            lr=(f"luminosity {L:.3g} suns, radius {R:.3g} suns"
                if L and R else
                f"radius {R:.3g} suns" if R else ""),
            life=f"main-sequence lifetime {t_ms:.2g} Gyr",
        ))
grid = pd.DataFrame(rows)

m_sel = alt.param(name="m_sel", value=0.0, bind=alt.binding_range(
    min=-1.0, max=2.45, step=0.05, name="log10 mass (suns)  "))
# 0.68 is on the age grid; the starter's 0.66 matched no row at load
a_sel = alt.param(name="a_sel", value=0.68, bind=alt.binding_range(
    min=-4.0, max=3.56, step=0.06, name="log10 age (Gyr)  "))
pick = ("abs(datum.lm - m_sel) < 0.02"
        " && abs(datum.la - a_sel) < 0.02")

# ---- the portrait ----------------------------------------------------
CX, CY = 160, 168
disc = alt.Chart(grid).transform_filter(pick).mark_circle(
    opacity=1).encode(
    x=alt.value(CX), y=alt.value(CY),
    size=alt.Size("size:Q", scale=None, legend=None),
    color=alt.Color("colour:N", scale=None, legend=None))
bh_ring = alt.Chart(grid).transform_filter(
    pick + ' && datum.phase == "black hole"').mark_point(
    filled=False, size=2400, stroke="#E07000", strokeWidth=3,
    opacity=1).encode(x=alt.value(CX), y=alt.value(CY))
# the Sun's size on the same log scale, for reference
sun_ring = alt.Chart(pd.DataFrame({"z": [0]})).mark_point(
    filled=False, size=int((14 + 26 * 2.2) ** 2),
    stroke="#DAA520", strokeWidth=1.5, opacity=1).encode(
    x=alt.value(CX), y=alt.value(CY))


def readout(field, y_px, size=12, color="#9aa1a8", bold=False):
    return alt.Chart(grid).transform_filter(pick).mark_text(
        fontSize=size, color=color,
        fontWeight="bold" if bold else "normal").encode(
        x=alt.value(CX), y=alt.value(y_px), text=field)


portrait = alt.layer(
    disc, bh_ring, sun_ring,
    readout("massage:N", 340),
    readout("phase:N", 364, size=15, color="#f5f2ea", bold=True),
    readout("temp:N", 386),
    readout("lr:N", 404),
    readout("life:N", 422),
).properties(width=320, height=440)

# ---- the HR diagram, with the star riding the sliders ----------------
# Temperature on x, luminosity on y, both logarithmic. By convention the
# temperature axis runs backwards, so hot sits on the left.
T_SCALE = alt.Scale(type="log", domain=[1200, 160000], reverse=True,
                    nice=False)
L_SCALE = alt.Scale(type="log", domain=[1e-4, 1e6], nice=False)
HX = alt.X("temp_K:Q", title="surface temperature (K), log scale",
           scale=T_SCALE,
           axis=alt.Axis(values=[2000, 3000, 5000, 10000, 20000, 50000,
                                 100000],
                         format=",", gridColor="#2b303b",
                         labelColor="#c8c8c8", titleColor="#c8c8c8"))
HY = alt.Y("lum:Q", title="luminosity (suns), log scale",
           scale=L_SCALE,
           axis=alt.Axis(values=[1e-4, 1e-2, 1, 1e2, 1e4, 1e6],
                         labelExpr="datum.value >= 1 ? "
                                   "format(datum.value, ',') : datum.value",
                         gridColor="#2b303b",
                         labelColor="#c8c8c8", titleColor="#c8c8c8"))

# the main sequence: every main-sequence mass at its own temperature and
# luminosity (L = M^3.5, T = SUN_T * M^0.475), one thick muted line
Mb = np.geomspace(0.1, 60, 200)
band = pd.DataFrame({"temp_K": SUN_T * Mb ** 0.475, "lum": Mb ** 3.5})
band_line = alt.Chart(band).mark_line(
    strokeWidth=9, strokeCap="round", color="#5a616b",
    opacity=0.6, clip=True).encode(x=HX, y=HY)

sun_hr = pd.DataFrame({"temp_K": [SUN_T], "lum": [1.0]})
sun_hr_pt = alt.Chart(sun_hr).mark_circle(
    size=55, color="#1e7d32", opacity=1).encode(x=HX, y=HY)
sun_hr_txt = alt.Chart(sun_hr).mark_text(
    align="left", dx=9, dy=12, fontSize=11, fontWeight=700,
    color="#1e7d32").encode(x=HX, y=HY, text=alt.value("Sun"))

# your star: the one grid row picked by the sliders. The grid holds each
# row's phase-adjusted temp_K and lum, so the point sits on the band while
# the star is on the main sequence and leaves it when the star does.
# Neutron stars, black holes, supernovae and "no remnant" have no lum,
# and the isValid test keeps them off the chart.
pick_hr = pick + " && isValid(datum.lum) && isValid(datum.temp_K)"
you = alt.Chart(grid).transform_filter(pick_hr).mark_point(
    shape=("M 0 -1 L 0.24 -0.31 L 0.95 -0.31 L 0.38 0.12 L 0.59 0.81"
           " L 0 0.38 L -0.59 0.81 L -0.38 0.12 L -0.95 -0.31"
           " L -0.24 -0.31 Z"),
    filled=True, size=280, color="#FFC300",
    stroke="#8C6A2F", strokeWidth=1.2, opacity=1, clip=True).encode(
    x=HX, y=HY)
you_label = alt.Chart(grid).transform_filter(pick_hr).mark_text(
    align="left", dx=14, fontSize=11, color="#f5f2ea",
    clip=True).encode(x=HX, y=HY, text="phase:N")

hr = alt.layer(
    band_line, sun_hr_pt, sun_hr_txt, you, you_label,
).properties(width=470, height=440,
             title=alt.Title("Hertzsprung-Russell diagram",
                             color="#f0f0f0"))

chart = alt.hconcat(portrait, hr).add_params(
    m_sel, a_sel).configure(background="#000000").configure_view(
    fill="#000000", stroke=None)

st.altair_chart(chart, use_container_width=False)
