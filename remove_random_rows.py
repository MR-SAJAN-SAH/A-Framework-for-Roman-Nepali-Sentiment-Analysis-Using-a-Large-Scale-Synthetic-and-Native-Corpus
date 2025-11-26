"""import pandas as pd

# Load your dataset
df = pd.read_csv("english.csv")

# Number of rows to remove
rows_to_remove = 186900

# Randomly sample rows to drop
rows_to_drop = df.sample(n=rows_to_remove, random_state=42).index

# Drop sampled rows
df_new = df.drop(rows_to_drop)

# Save the new dataset
df_new.to_csv("EnglishData.csv", index=False)

print("Original rows:", len(df))
print("New rows:", len(df_new))
print("Removed:", len(df) - len(df_new))


"""

import pandas as pd

# Load CSV
df = pd.read_csv("EnglishData.csv")   # replace with your file name

df["sentiment"].isna().any()
df = df.dropna(subset=["sentiment"])

df["sentiment"].isna().sum()
print("NaN count:", df["sentiment"].isna().sum())
print("Non-NaN count:", df["sentiment"].notna().sum())


# Save to new CSV
df.to_csv("English.csv", index=False)
