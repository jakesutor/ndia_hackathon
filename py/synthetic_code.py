# Generate a synthetic Cerner-like encounter dataset for USAF active-duty pilots (last 5 years)
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import string
import json
import os

rng = np.random.default_rng(42)
today = datetime(2025, 8, 25)

# -----------------------
# Helpers
# -----------------------
def random_dates(start, end, n):
    """Return sorted list of n random datetimes between start and end."""
    start_u = start.timestamp()
    end_u = end.timestamp()
    return sorted([datetime.fromtimestamp(rng.uniform(start_u, end_u)) for _ in range(n)])

def pick_weighted(options, probs):
    return rng.choice(options, p=np.array(probs)/np.sum(probs))

def idgen(prefix, n):
    return [prefix + ''.join(rng.choice(list(string.ascii_uppercase + string.digits), size=8)) for _ in range(n)]

def sample_icd10():
    codes = [
        ("H52.1","Myopia"),
        ("J30.9","Allergic rhinitis, unspecified"),
        ("G47.00","Insomnia, unspecified"),
        ("I10","Essential (primary) hypertension"),
        ("M54.5","Low back pain"),
        ("S06.0X0A","Concussion without loss of consciousness, initial"),
        ("F41.1","Generalized anxiety disorder"),
        ("H81.10","Benign paroxysmal vertigo, unspecified ear"),
        ("R07.9","Chest pain, unspecified"),
        ("K35.80","Unspecified acute appendicitis")
    ]
    return codes[rng.integers(len(codes))]

def sample_cpt():
    codes = [
        ("92015","Determination of refractive state"),
        ("93000","Electrocardiogram; complete"),
        ("81001","Urinalysis, automated, with microscopy"),
        ("92557","Comprehensive audiometry"),
        ("85025","Complete CBC, automated"),
        ("80053","Comprehensive metabolic panel"),
        ("80061","Lipid panel"),
        ("99213","Office/outpatient visit est"),
        ("99283","ED visit, level 3")
    ]
    return codes[rng.integers(len(codes))]

def sample_med():
    meds = ["Loratadine 10mg", "Melatonin 3mg", "Lisinopril 10mg", "Ibuprofen 400mg", "Sertraline 50mg", "None"]
    probs = [0.18,0.10,0.06,0.22,0.05,0.39]
    return rng.choice(meds, p=probs)

bases = [
    "Hill AFB", "Langley AFB", "Luke AFB", "Nellis AFB", "Eglin AFB",
    "Joint Base San Antonio", "Joint Base Elmendorf-Richardson", "Osan AB",
    "Misawa AB", "RAF Lakenheath"
]

aircraft = ["F-16","F-22","F-35","KC-135","C-17","HH-60","T-38","A-10"]
ranks = ["O-1","O-2","O-3","O-4","O-5","O-6"]
visit_types = ["Primary Care","Flight Medicine","ENT","Cardiology","Urgent Care","Behavioral Health","Emergency Dept"]

# -----------------------
# Pilot master table
# -----------------------
n_pilots = 500
pilot_ids = [f"P{str(i).zfill(4)}" for i in range(1, n_pilots+1)]
ages = rng.integers(25, 46, size=n_pilots)
genders = rng.choice(["Male","Female"], size=n_pilots, p=[0.9,0.1])
pilot_ranks = rng.choice(ranks, size=n_pilots, p=[0.12,0.18,0.30,0.24,0.12,0.04])
pilot_aircraft = rng.choice(aircraft, size=n_pilots, p=[0.18,0.08,0.22,0.12,0.15,0.08,0.10,0.07])
pilot_bases = rng.choice(bases, size=n_pilots)
total_hours = rng.integers(500, 4001, size=n_pilots)
hours_12mo = np.clip((rng.normal(250, 70, size=n_pilots)).astype(int), 50, 600)

aero_classes = rng.choice(["Class I","Class II","Class III"], size=n_pilots, p=[0.7,0.25,0.05])
dental_readiness = rng.choice(["1","2","3"], size=n_pilots, p=[0.75,0.2,0.05])

# PHA
pha_last_dates = [today - timedelta(days=int(x)) for x in rng.integers(0, 500, size=n_pilots)]
pha_status = ["Complete" if (today - d).days < 365 else "Overdue" for d in pha_last_dates]

pilot_df = pd.DataFrame({
    "pilot_id": pilot_ids,
    "age": ages,
    "gender": genders,
    "rank": pilot_ranks,
    "base": pilot_bases,
    "aircraft": pilot_aircraft,
    "flight_hours_total": total_hours,
    "flight_hours_last_12mo": hours_12mo,
    "aeromedical_class_current": aero_classes,
    "dental_readiness": dental_readiness,
    "pha_last_date": pha_last_dates,
    "pha_status": pha_status
})

# -----------------------
# Immunizations
# -----------------------
vaccines = ["Influenza","COVID-19","Hepatitis A","Hepatitis B","Tdap","MMR"]
imm_rows = []
for pid in pilot_ids:
    for v in vaccines:
        n_shots = 1 if v in ["Influenza","Tdap"] else rng.integers(1, 3)
        dates = random_dates(datetime(2020,1,1), today, n_shots)
        for dt in dates:
            imm_rows.append({
                "imm_id": f"IMM{pid}{dt.strftime('%Y%m%d')}",
                "pilot_id": pid,
                "vaccine": v,
                "date": dt.date().isoformat(),
                "status": "Completed"
            })
immun_df = pd.DataFrame(imm_rows)

# -----------------------
# Aeromedical classification history
# -----------------------
aero_rows = []
for pid, current_cls in zip(pilot_ids, aero_classes):
    # 1-4 records over time
    n_rec = rng.integers(1,5)
    dates = random_dates(datetime(2020,1,1), today, n_rec)
    for i, dt in enumerate(dates):
        # Mostly stable, occasional temporary change
        cls = current_cls if i == n_rec-1 or rng.random() < 0.8 else rng.choice(["Class I","Class II","Class III"], p=[0.6,0.3,0.1])
        temp_flag = bool(rng.random() < 0.12)
        aero_rows.append({
            "record_id": f"AR{pid}{i}",
            "pilot_id": pid,
            "date": dt.date().isoformat(),
            "classification": cls,
            "temporary_profile_flag": temp_flag
        })
aero_hist_df = pd.DataFrame(aero_rows)

# -----------------------
# Profiles (flight duty restricting)
# -----------------------
profile_reasons = ["Back pain","Elevated BP","Concussion eval","Vision changes","Post-op limitation","Anxiety management"]
profile_rows = []
for pid in pilot_ids:
    # chance of having profiles
    if rng.random() < 0.35:
        n_prof = rng.integers(1,4)
        starts = random_dates(datetime(2020,1,1), today - timedelta(days=7), n_prof)
        for i, st in enumerate(starts):
            dur = rng.integers(7, 90)
            end = st + timedelta(days=int(dur))
            profile_rows.append({
                "profile_id": f"PR{pid}{i}",
                "pilot_id": pid,
                "start_date": st.date().isoformat(),
                "end_date": min(end, today).date().isoformat(),
                "type": rng.choice(["Temporary","Permanent"], p=[0.9,0.1]),
                "reason": rng.choice(profile_reasons)
            })
profiles_df = pd.DataFrame(profile_rows)

# -----------------------
# Encounters
# -----------------------
enc_rows = []
enc_cnts = rng.integers(2, 15, size=n_pilots)
eid = 0
for pid, cnt in zip(pilot_ids, enc_cnts):
    dates = random_dates(datetime(2020,1,1), today, int(cnt))
    for dt in dates:
        icd, icd_desc = sample_icd10()
        cpt, cpt_desc = sample_cpt()
        vt = rng.choice(visit_types, p=[0.25,0.25,0.08,0.07,0.15,0.12,0.08])
        enc_rows.append({
            "encounter_id": f"ENC{eid:07d}",
            "pilot_id": pid,
            "date": dt.date().isoformat(),
            "visit_type": vt,
            "icd10": icd,
            "icd10_desc": icd_desc,
            "cpt": cpt,
            "cpt_desc": cpt_desc,
            "medication": sample_med()
        })
        eid += 1
encounters_df = pd.DataFrame(enc_rows)

# -----------------------
# Labs
# -----------------------
lab_tests = [
    ("CBC","Hemoglobin","g/dL",13.5,17.5),
    ("CBC","WBC","10^3/uL",4.0,11.0),
    ("CBC","Platelets","10^3/uL",150,450),
    ("CMP","Glucose","mg/dL",70,99),
    ("CMP","Creatinine","mg/dL",0.7,1.3),
    ("CMP","ALT","U/L",7,56),
    ("Lipid","LDL","mg/dL",0,100),
    ("Lipid","HDL","mg/dL",40,200),
    ("Lipid","Triglycerides","mg/dL",0,150)
]

lab_rows = []
lid = 0
for pid in pilot_ids:
    # 1-6 lab panels over time
    n_draws = rng.integers(1,7)
    draw_dates = random_dates(datetime(2020,1,1), today, int(n_draws))
    for dd in draw_dates:
        for panel, test, unit, lo, hi in lab_tests:
            # sample with slight chance of abnormal
            if rng.random() < 0.12:
                # push value outside range
                if rng.random() < 0.5:
                    val = hi + abs(rng.normal(0, (hi-lo)*0.15))
                else:
                    val = max(0, lo - abs(rng.normal(0, (hi-lo)*0.15)))
            else:
                val = rng.normal((lo+hi)/2, (hi-lo)/8)
            flag = "H" if val > hi else ("L" if val < lo else "N")
            lab_rows.append({
                "lab_id": f"LAB{lid:07d}",
                "pilot_id": pid,
                "date": dd.date().isoformat(),
                "panel": panel,
                "test_name": test,
                "result_value": round(float(val),2),
                "unit": unit,
                "ref_low": lo,
                "ref_high": hi,
                "abnormal_flag": flag
            })
            lid += 1
labs_df = pd.DataFrame(lab_rows)

# -----------------------
# Mental health
# -----------------------
mh_dx = ["Adjustment disorder","Anxiety disorder","Insomnia","None"]
mh_probs = [0.08,0.07,0.10,0.75]
mh_meds = ["Sertraline","Escitalopram","Trazodone","None"]
mh_rows = []
mid = 0
for pid in pilot_ids:
    if rng.random() < 0.35:
        n_mh = rng.integers(1,5)
        dates = random_dates(datetime(2020,1,1), today, int(n_mh))
        for i, dt in enumerate(dates):
            dx = rng.choice(mh_dx, p=mh_probs)
            med = "None" if dx=="None" else rng.choice(mh_meds, p=[0.3,0.3,0.2,0.2])
            sessions = 0 if dx=="None" else rng.integers(1,8)
            mh_rows.append({
                "mh_id": f"MH{mid:07d}",
                "pilot_id": pid,
                "date": dt.date().isoformat(),
                "diagnosis": dx,
                "medication": med,
                "therapy_sessions": int(sessions)
            })
            mid += 1
mh_df = pd.DataFrame(mh_rows)

# -----------------------
# Hospitalizations / ED
# -----------------------
hosp_reasons = ["Appendicitis","Fracture repair","Chest pain obs","Pneumonia","Dehydration","None"]
hosp_rows = []
hid = 0
for pid in pilot_ids:
    if rng.random() < 0.15:
        n_h = rng.integers(1,3)
        admits = random_dates(datetime(2020,1,1), today - timedelta(days=1), int(n_h))
        for ad in admits:
            los = rng.integers(1,7)
            dc = ad + timedelta(days=int(los))
            hosp_rows.append({
                "hosp_id": f"H{hid:07d}",
                "pilot_id": pid,
                "admit_date": ad.date().isoformat(),
                "discharge_date": min(dc, today).date().isoformat(),
                "reason": rng.choice(hosp_reasons, p=[0.25,0.15,0.25,0.15,0.15,0.05]),
                "via_ed": bool(rng.random() < 0.6)
            })
            hid += 1
hosp_df = pd.DataFrame(hosp_rows)

# -----------------------
# Readiness monthly KPIs (by base)
# -----------------------
def month_range(start, end):
    d = datetime(start.year, start.month, 1)
    months = []
    while d <= end:
        months.append(d)
        # next month
        if d.month == 12:
            d = datetime(d.year+1,1,1)
        else:
            d = datetime(d.year, d.month+1, 1)
    return months

months = month_range(datetime(2020,1,1), today)
kpi_rows = []
for b in bases:
    for m in months:
        # base size proxy
        base_size = max(20, int(rng.normal(50, 10)))
        pha_rate = np.clip(rng.normal(0.86, 0.06), 0.6, 0.99)
        flight_phys_rate = np.clip(rng.normal(0.90, 0.05), 0.7, 1.0)
        profile_rate = np.clip(rng.normal(0.08, 0.03), 0.0, 0.3)
        immun_rate = np.clip(rng.normal(0.88, 0.07), 0.6, 1.0)
        dental_rate = np.clip(rng.normal(0.92, 0.04), 0.7, 1.0)
        kpi_rows.append({
            "base": b,
            "year_month": m.strftime("%Y-%m"),
            "population": base_size,
            "pha_completion_rate": round(float(pha_rate),3),
            "flight_physical_compliance": round(float(flight_phys_rate),3),
            "profile_rate": round(float(profile_rate),3),
            "immunization_compliance": round(float(immun_rate),3),
            "dental_readiness_rate": round(float(dental_rate),3)
        })
kpi_df = pd.DataFrame(kpi_rows)

# -----------------------
# Predictive labels (monthly per pilot)
# -----------------------
label_rows = []
for pid in pilot_ids:
    # choose 12 recent months
    months_recent = months[-18:]
    m_sel = rng.choice(months_recent, size=12, replace=False)
    for m in sorted(m_sel):
        # label: did pilot get a new profile in next 90 days?
        future_profiles = profiles_df[
            (profiles_df["pilot_id"]==pid) &
            (pd.to_datetime(profiles_df["start_date"])>m) &
            (pd.to_datetime(profiles_df["start_date"])<= (m + timedelta(days=90)))
        ]
        label = 1 if len(future_profiles)>0 else 0
        # simple feature proxies
        fh12 = pilot_df.loc[pilot_df["pilot_id"]==pid,"flight_hours_last_12mo"].iloc[0]
        age = pilot_df.loc[pilot_df["pilot_id"]==pid,"age"].iloc[0]
        # recent abnormal labs count
        recent_labs = labs_df[(labs_df["pilot_id"]==pid) & (pd.to_datetime(labs_df["date"])> (m - timedelta(days=180)))]
        abn = int((recent_labs["abnormal_flag"]!="N").sum())
        # recent diagnoses count
        recent_dx = encounters_df[(encounters_df["pilot_id"]==pid) & (pd.to_datetime(encounters_df["date"])> (m - timedelta(days=180)))]
        dx_count = int(recent_dx.shape[0])
        # recent PHA status
        pha = pilot_df.loc[pilot_df["pilot_id"]==pid,"pha_status"].iloc[0]
        pha_flag = 0 if pha=="Complete" else 1
        label_rows.append({
            "pilot_id": pid,
            "year_month": m.strftime("%Y-%m"),
            "age": int(age),
            "flight_hours_last_12mo": int(fh12),
            "abnormal_labs_6mo": int(abn),
            "encounters_6mo": int(dx_count),
            "pha_overdue_flag": int(pha_flag),
            "profile_next_90d": int(label)
        })
labels_df = pd.DataFrame(label_rows)

# -----------------------
# Save all to CSV and XLSX
# -----------------------
outdir = "/mnt/data/usaf_pilots_synthetic"
os.makedirs(outdir, exist_ok=True)

datasets = {
    "pilots.csv": pilot_df,
    "immunizations.csv": immun_df,
    "aeromedical_class_history.csv": aero_hist_df,
    "profiles.csv": profiles_df,
    "encounters.csv": encounters_df,
    "labs.csv": labs_df,
    "mental_health.csv": mh_df,
    "hospitalizations.csv": hosp_df,
    "readiness_kpis_monthly.csv": kpi_df,
    "predictive_labels_monthly.csv": labels_df
}

for name, df in datasets.items():
    df.to_csv(os.path.join(outdir, name), index=False)

# Also produce a single Excel workbook
xlsx_path = os.path.join(outdir, "usaf_pilots_synthetic.xlsx")
with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as xw:
    for name, df in datasets.items():
        sheet = name.replace(".csv","")[:31]
        df.to_excel(xw, sheet_name=sheet, index=False)

# -----------------------
# Create a ready-to-run Python dashboard demo script
# -----------------------
demo_code = r'''
import pandas as pd
import matplotlib.pyplot as plt

# Load
base = "usaf_pilots_synthetic"
pilots = pd.read_csv(f"{base}/pilots.csv", parse_dates=["pha_last_date"])
kpis = pd.read_csv(f"{base}/readiness_kpis_monthly.csv")
profiles = pd.read_csv(f"{base}/profiles.csv", parse_dates=["start_date","end_date"])
labels = pd.read_csv(f"{base}/predictive_labels_monthly.csv")
enc = pd.read_csv(f"{base}/encounters.csv", parse_dates=["date"])
aero = pd.read_csv(f"{base}/aeromedical_class_history.csv", parse_dates=["date"])

# KPIs
total_pilots = pilots.shape[0]
pha_complete = (pilots["pha_status"]=="Complete").mean()
profile_rate = (profiles["pilot_id"].nunique() / total_pilots) if total_pilots else 0.0

print("=== KPI Snapshot ===")
print(f"Pilots: {total_pilots}")
print(f"PHA Completion Rate: {pha_complete:.2%}")
print(f"Any Profile (lifetime): {profile_rate:.2%}")

# Trend: profile rate over time (from KPIs baseline)
base_kpi = kpis.groupby("year_month")["profile_rate"].mean().reset_index()
plt.figure()
plt.plot(base_kpi["year_month"], base_kpi["profile_rate"])
plt.xticks(rotation=60)
plt.title("Average Profile Rate Over Time")
plt.xlabel("Month")
plt.ylabel("Profile Rate")
plt.tight_layout()
plt.show()

# Distribution: Aeromedical classification
latest_aero = aero.sort_values("date").groupby("pilot_id").tail(1)
dist = latest_aero["classification"].value_counts().reset_index()
plt.figure()
plt.bar(dist["index"], dist["classification"])
plt.title("Aeromedical Classification Distribution (Latest)")
plt.xlabel("Class")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Age distribution
plt.figure()
pilots["age"].plot(kind="hist", bins=15)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Simple predictive baseline (logistic regression as example)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report

X = labels[["age","flight_hours_last_12mo","abnormal_labs_6mo","encounters_6mo","pha_overdue_flag"]]
y = labels["profile_next_90d"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=7)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
probs = clf.predict_proba(X_test)[:,1]
print("AUC:", roc_auc_score(y_test, probs))
print(classification_report(y_test, (probs>0.5).astype(int)))
'''
demo_path = os.path.join(outdir, "dashboard_demo.py")
with open(demo_path, "w") as f:
    f.write(demo_code)

# Provide a compact preview to the user
import caas_jupyter_tools as cj
cj.display_dataframe_to_user("Preview: pilots.csv (first 25 rows)", pilot_df.head(25))

outdir, xlsx_path, list(datasets.keys()), demo_path
