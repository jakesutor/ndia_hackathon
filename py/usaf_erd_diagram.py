# Create an Entity Relationship Diagram (ERD) as PNG and SVG using matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import matplotlib as mpl

mpl.rcParams["figure.dpi"] = 160

# Canvas
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 12)
ax.set_ylim(0, 9)
ax.axis('off')

def draw_table(x, y, w, h, title, fields, color="#F5F7FB"):
    # Outer box
    rect = Rectangle((x, y), w, h, linewidth=1.5, edgecolor="#4B5563", facecolor=color, zorder=1)
    ax.add_patch(rect)
    # Title bar
    ax.add_patch(Rectangle((x, y+h-0.6), w, 0.6, linewidth=0, facecolor="#E5E7EB", zorder=2))
    ax.text(x+0.2, y+h-0.35, title, fontsize=11, fontweight="bold", va="center", ha="left", color="#111827", zorder=3)
    # Fields
    for i, f in enumerate(fields):
        ax.text(x+0.2, y+h-0.9-0.4*i, f, fontsize=9, va="top", ha="left", color="#111827", zorder=3)

def connect_tables(p1, p2, text="1..N", style="solid"):
    # p1, p2 are (x, y) anchor centers
    arrow = FancyArrowPatch(p1, p2, arrowstyle="-", mutation_scale=10,
                            linewidth=1.2, color="#374151", linestyle=style, zorder=0)
    ax.add_patch(arrow)
    # cardinalities near ends
    ax.text((p1[0]*0.7+p2[0]*0.3), (p1[1]*0.7+p2[1]*0.3)+0.1, "1", fontsize=9, color="#111827")
    ax.text((p1[0]*0.3+p2[0]*0.7), (p1[1]*0.3+p2[1]*0.7)-0.1, "N", fontsize=9, color="#111827")

# Define tables
tables = {}

tables["pilots"] = {
    "pos": (1.0, 5.8), "size": (3.7, 3.0),
    "fields": ["pilot_id (PK)",
               "age, gender, rank",
               "base, aircraft",
               "flight_hours_total",
               "flight_hours_last_12mo",
               "aeromedical_class_current",
               "dental_readiness",
               "pha_last_date, pha_status"]
}

tables["encounters"] = {
    "pos": (6.0, 7.0), "size": (3.8, 2.2),
    "fields": ["encounter_id (PK)",
               "pilot_id (FK)",
               "date, visit_type",
               "icd10, icd10_desc",
               "cpt, cpt_desc, medication"]
}

tables["labs"] = {
    "pos": (6.0, 4.3), "size": (3.8, 2.2),
    "fields": ["lab_id (PK)",
               "pilot_id (FK)",
               "date, panel, test_name",
               "result_value, unit",
               "ref_low, ref_high, abnormal_flag"]
}

tables["immunizations"] = {
    "pos": (6.0, 1.6), "size": (3.8, 2.0),
    "fields": ["imm_id (PK)",
               "pilot_id (FK)",
               "vaccine, date, status"]
}

tables["aero_hist"] = {
    "pos": (10.0, 7.0), "size": (3.8, 2.0),
    "fields": ["record_id (PK)",
               "pilot_id (FK)",
               "date, classification",
               "temporary_profile_flag"]
}

tables["profiles"] = {
    "pos": (10.0, 4.3), "size": (3.8, 2.2),
    "fields": ["profile_id (PK)",
               "pilot_id (FK)",
               "start_date, end_date",
               "type, reason"]
}

tables["mental_health"] = {
    "pos": (10.0, 1.6), "size": (3.8, 2.0),
    "fields": ["mh_id (PK)",
               "pilot_id (FK)",
               "date, diagnosis",
               "medication, therapy_sessions"]
}

tables["hospitalizations"] = {
    "pos": (1.0, 2.3), "size": (3.7, 2.0),
    "fields": ["hosp_id (PK)",
               "pilot_id (FK)",
               "admit_date, discharge_date",
               "reason, via_ed"]
}

tables["readiness_kpis_monthly"] = {
    "pos": (1.0, 0.4), "size": (5.0, 1.6),
    "fields": ["base, year_month (grain)",
               "population",
               "pha_completion_rate",
               "flight_physical_compliance",
               "profile_rate",
               "immunization_compliance",
               "dental_readiness_rate"]
}

tables["predictive_labels_monthly"] = {
    "pos": (5.6, 0.4), "size": (5.4, 1.6),
    "fields": ["pilot_id (FK), year_month (grain)",
               "age, flight_hours_last_12mo",
               "abnormal_labs_6mo, encounters_6mo",
               "pha_overdue_flag, profile_next_90d (label)"]
}

# Draw tables
for name, meta in tables.items():
    x, y = meta["pos"]
    w, h = meta["size"]
    title = name
    draw_table(x, y, w, h, title, meta["fields"])

# Connect relationships (1 to N from pilots)
anchor = lambda meta: (meta["pos"][0]+meta["size"][0]/2, meta["pos"][1]+meta["size"][1]/2)

one = anchor(tables["pilots"])
for rel in ["encounters","labs","immunizations","aero_hist","profiles","mental_health","hospitalizations","predictive_labels_monthly"]:
    connect_tables(one, anchor(tables[rel]))

# Dotted association between pilots.base and readiness_kpis_monthly.base (not a strict FK in synthetic set)
arrow = FancyArrowPatch(anchor(tables["pilots"]), anchor(tables["readiness_kpis_monthly"]), 
                        arrowstyle="-", mutation_scale=10, linewidth=1.2, 
                        color="#374151", linestyle="dashed", zorder=0)
ax.add_patch(arrow)
ax.text(2.9, 2.6, "base (assoc.)", fontsize=9, color="#111827")

# Title
ax.text(0.3, 8.7, "USAF Pilot Synthetic Cerner/MHS GENESIS ERD", fontsize=14, fontweight="bold", color="#111827")

png_path = "/mnt/data/usaf_pilots_ERD.png"
svg_path = "/mnt/data/usaf_pilots_ERD.svg"
fig.savefig(png_path, bbox_inches="tight")
fig.savefig(svg_path, bbox_inches="tight")
png_path, svg_path
