import os
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_CSV = os.path.join("data", "Labels.csv")
SPLIT_DIR = "splits"
RANDOM_STATE = 42

os.makedirs(SPLIT_DIR, exist_ok=True)

df = pd.read_csv(DATA_CSV)

required_cols = ["Image Name", "Patient", "Label"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Kolom '{col}' tidak ditemukan di Labels.csv")

# Ambil satu label per patient untuk stratified patient-level split
patient_df = df.groupby("Patient")["Label"].first().reset_index()

train_pat, temp_pat = train_test_split(
    patient_df,
    test_size=0.30,
    stratify=patient_df["Label"],
    random_state=RANDOM_STATE
)

val_pat, test_pat = train_test_split(
    temp_pat,
    test_size=0.50,
    stratify=temp_pat["Label"],
    random_state=RANDOM_STATE
)

def filter_df_by_patients(full_df, patient_subset):
    patient_ids = patient_subset["Patient"].tolist()
    return full_df[full_df["Patient"].isin(patient_ids)].copy()

train_df = filter_df_by_patients(df, train_pat)
val_df = filter_df_by_patients(df, val_pat)
test_df = filter_df_by_patients(df, test_pat)

train_df.to_csv(os.path.join(SPLIT_DIR, "train.csv"), index=False)
val_df.to_csv(os.path.join(SPLIT_DIR, "val.csv"), index=False)
test_df.to_csv(os.path.join(SPLIT_DIR, "test.csv"), index=False)

print("Split selesai.")
print(f"Train images: {len(train_df)} | patients: {train_df['Patient'].nunique()}")
print(f"Val images:   {len(val_df)} | patients: {val_df['Patient'].nunique()}")
print(f"Test images:  {len(test_df)} | patients: {test_df['Patient'].nunique()}")