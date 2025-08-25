import os
import pandas as pd

# Prepare dictionaries for each dataset
dicts = {}

dicts["pilots"] = pd.DataFrame([
    {"Column":"pilot_id","Type":"string (PK)","Example":"P0007","Description":"Unique pilot identifier. Primary key used across datasets."},
    {"Column":"age","Type":"int","Example":"33","Description":"Age in years."},
    {"Column":"gender","Type":"string","Example":"Male","Description":"Pilot gender. Values: Male, Female."},
    {"Column":"rank","Type":"string","Example":"O-3","Description":"Officer rank (O-1…O-6)."},
    {"Column":"base","Type":"string","Example":"Nellis AFB","Description":"Assigned base."},
    {"Column":"aircraft","Type":"string","Example":"F-35","Description":"Primary aircraft type."},
    {"Column":"flight_hours_total","Type":"int","Example":"2140","Description":"Cumulative flight hours."},
    {"Column":"flight_hours_last_12mo","Type":"int","Example":"280","Description":"Flight hours in the last 12 months."},
    {"Column":"aeromedical_class_current","Type":"string","Example":"Class I","Description":"Current aeromedical class (Class I, II, III)."},
    {"Column":"dental_readiness","Type":"string","Example":"1","Description":"Dental readiness class (1=deployable,2=deployable treatment due,3=non-deployable)."},
    {"Column":"pha_last_date","Type":"date","Example":"2025-04-19","Description":"Date of last PHA."},
    {"Column":"pha_status","Type":"string","Example":"Complete","Description":"PHA status (Complete or Overdue)."},
])

dicts["immunizations"] = pd.DataFrame([
    {"Column":"imm_id","Type":"string (PK)","Example":"IMMP000720230930","Description":"Unique immunization record ID."},
    {"Column":"pilot_id","Type":"string (FK)","Example":"P0007","Description":"Pilot foreign key."},
    {"Column":"vaccine","Type":"string","Example":"Influenza","Description":"Vaccine name."},
    {"Column":"date","Type":"date","Example":"2023-09-30","Description":"Administration date."},
    {"Column":"status","Type":"string","Example":"Completed","Description":"Immunization status."},
])

dicts["aeromedical_class_history"] = pd.DataFrame([
    {"Column":"record_id","Type":"string (PK)","Example":"ARP00071","Description":"Unique record ID."},
    {"Column":"pilot_id","Type":"string (FK)","Example":"P0007","Description":"Pilot foreign key."},
    {"Column":"date","Type":"date","Example":"2024-06-12","Description":"Classification date."},
    {"Column":"classification","Type":"string","Example":"Class I","Description":"Aeromedical class."},
    {"Column":"temporary_profile_flag","Type":"boolean","Example":"true","Description":"Flag for temporary flight duty restriction."},
])

dicts["profiles"] = pd.DataFrame([
    {"Column":"profile_id","Type":"string (PK)","Example":"PRP00071","Description":"Unique profile ID."},
    {"Column":"pilot_id","Type":"string (FK)","Example":"P0007","Description":"Pilot foreign key."},
    {"Column":"start_date","Type":"date","Example":"2024-10-02","Description":"Profile start date."},
    {"Column":"end_date","Type":"date","Example":"2024-10-30","Description":"Profile end date."},
    {"Column":"type","Type":"string","Example":"Temporary","Description":"Profile type (Temporary or Permanent)."},
    {"Column":"reason","Type":"string","Example":"Elevated BP","Description":"Reason for profile."},
])

dicts["encounters"] = pd.DataFrame([
    {"Column":"encounter_id","Type":"string (PK)","Example":"ENC0000123","Description":"Unique encounter ID."},
    {"Column":"pilot_id","Type":"string (FK)","Example":"P0007","Description":"Pilot foreign key."},
    {"Column":"date","Type":"date","Example":"2023-03-15","Description":"Encounter date."},
    {"Column":"visit_type","Type":"string","Example":"Flight Medicine","Description":"Visit type."},
    {"Column":"icd10","Type":"string","Example":"I10","Description":"ICD-10 code."},
    {"Column":"icd10_desc","Type":"string","Example":"Essential hypertension","Description":"ICD-10 description."},
    {"Column":"cpt","Type":"string","Example":"93000","Description":"CPT code."},
    {"Column":"cpt_desc","Type":"string","Example":"ECG complete","Description":"CPT description."},
    {"Column":"medication","Type":"string","Example":"Lisinopril 10mg","Description":"Medication prescribed."},
])

dicts["labs"] = pd.DataFrame([
    {"Column":"lab_id","Type":"string (PK)","Example":"LAB0000456","Description":"Unique lab record ID."},
    {"Column":"pilot_id","Type":"string (FK)","Example":"P0007","Description":"Pilot foreign key."},
    {"Column":"date","Type":"date","Example":"2024-02-11","Description":"Lab draw date."},
    {"Column":"panel","Type":"string","Example":"Lipid","Description":"Panel name."},
    {"Column":"test_name","Type":"string","Example":"LDL","Description":"Test name."},
    {"Column":"result_value","Type":"float","Example":"121.0","Description":"Result value."},
    {"Column":"unit","Type":"string","Example":"mg/dL","Description":"Units of measure."},
    {"Column":"ref_low","Type":"float","Example":"0.0","Description":"Reference low."},
    {"Column":"ref_high","Type":"float","Example":"100.0","Description":"Reference high."},
    {"Column":"abnormal_flag","Type":"string","Example":"H","Description":"Flag: N normal, H high, L low."},
])

dicts["mental_health"] = pd.DataFrame([
    {"Column":"mh_id","Type":"string (PK)","Example":"MH0000321","Description":"Unique mental health record ID."},
    {"Column":"pilot_id","Type":"string (FK)","Example":"P0007","Description":"Pilot foreign key."},
    {"Column":"date","Type":"date","Example":"2023-07-05","Description":"MH encounter date."},
    {"Column":"diagnosis","Type":"string","Example":"Anxiety disorder","Description":"Diagnosis."},
    {"Column":"medication","Type":"string","Example":"Sertraline","Description":"Medication."},
    {"Column":"therapy_sessions","Type":"int","Example":"4","Description":"Number of sessions."},
])

dicts["hospitalizations"] = pd.DataFrame([
    {"Column":"hosp_id","Type":"string (PK)","Example":"H0000102","Description":"Unique hospitalization ID."},
    {"Column":"pilot_id","Type":"string (FK)","Example":"P0007","Description":"Pilot foreign key."},
    {"Column":"admit_date","Type":"date","Example":"2022-11-12","Description":"Admission date."},
    {"Column":"discharge_date","Type":"date","Example":"2022-11-16","Description":"Discharge date."},
    {"Column":"reason","Type":"string","Example":"Chest pain obs","Description":"Reason for hospitalization."},
    {"Column":"via_ed","Type":"boolean","Example":"true","Description":"Admission via ED flag."},
])

dicts["readiness_kpis_monthly"] = pd.DataFrame([
    {"Column":"base","Type":"string","Example":"Langley AFB","Description":"Base name."},
    {"Column":"year_month","Type":"string (YYYY-MM)","Example":"2023-10","Description":"Month."},
    {"Column":"population","Type":"int","Example":"52","Description":"Pilot population."},
    {"Column":"pha_completion_rate","Type":"float","Example":"0.903","Description":"PHA completion rate (0-1)."},
    {"Column":"flight_physical_compliance","Type":"float","Example":"0.927","Description":"Flight physical compliance rate."},
    {"Column":"profile_rate","Type":"float","Example":"0.081","Description":"Profile rate (0-1)."},
    {"Column":"immunization_compliance","Type":"float","Example":"0.884","Description":"Immunization compliance rate (0-1)."},
    {"Column":"dental_readiness_rate","Type":"float","Example":"0.941","Description":"Dental readiness rate (0-1)."},
])

dicts["predictive_labels_monthly"] = pd.DataFrame([
    {"Column":"pilot_id","Type":"string (FK)","Example":"P0007","Description":"Pilot foreign key."},
    {"Column":"year_month","Type":"string (YYYY-MM)","Example":"2025-03","Description":"Reference month."},
    {"Column":"age","Type":"int","Example":"33","Description":"Pilot age."},
    {"Column":"flight_hours_last_12mo","Type":"int","Example":"280","Description":"Flight hours last 12 months."},
    {"Column":"abnormal_labs_6mo","Type":"int","Example":"2","Description":"Abnormal labs in past 6 months."},
    {"Column":"encounters_6mo","Type":"int","Example":"5","Description":"Encounters in past 6 months."},
    {"Column":"pha_overdue_flag","Type":"int (0/1)","Example":"0","Description":"PHA overdue flag."},
    {"Column":"profile_next_90d","Type":"int (0/1)","Example":"1","Description":"Label: profile in next 90 days."},
])

# Write to Excel workbook
outpath = "/mnt/data/usaf_pilots_data_dictionary.xlsx"
with pd.ExcelWriter(outpath, engine="xlsxwriter") as writer:
    for name, df in dicts.items():
        df.to_excel(writer, sheet_name=name[:31], index=False)

outpath
