
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
