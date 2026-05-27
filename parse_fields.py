import numpy as np
import pandas as pd
import json

xlxs_file = "SEDCIP24_TACC.xlsx"

df = pd.read_excel(xlxs_file, sheet_name="SEDCIP24", header=0)

drop_cols = [
    "SED_CIPCode", "CIP Code", "Broad_Field_Code", "Major_Field_code",
    "Detailed_Field_Code", "Trend_Broad_Field_Code", "Trend_Major_Field_Code",
    "End_year", "Active", "Field_Area", "Trend_Broad_Field_Label",
    "Trend_Major_Field_Label", "Field_Area_Label"
]
df = df.drop(columns=drop_cols)

num_records = len(df)
print(f"Read {num_records} records from {xlxs_file}")

df.to_json("SEDCIP24.json", orient="records", indent=2)

# Write unique values for each field level to separate JSON files
for col, filename in [
    ("Broad_Field_label", "broad_fields.json"),
    ("Major_Field_label", "major_fields.json"),
    ("Detailed_Field_label", "detailed_fields.json"),
]:
    values = sorted(df[col].dropna().unique().tolist())
    with open(filename, "w") as f:
        json.dump(values, f, indent=2)
    print(f"Wrote {len(values)} unique values to {filename}")
