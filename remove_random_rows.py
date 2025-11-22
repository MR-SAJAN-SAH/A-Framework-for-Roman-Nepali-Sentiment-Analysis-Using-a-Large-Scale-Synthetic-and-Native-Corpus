import pandas as pd

# Load your dataset
df = pd.read_csv("Nepali.csv")

# Number of rows to remove
rows_to_remove = 15788

# Randomly sample rows to drop
rows_to_drop = df.sample(n=rows_to_remove, random_state=42).index

# Drop sampled rows
df_new = df.drop(rows_to_drop)

# Save the new dataset
df_new.to_csv("Nepali1.csv", index=False)

print("Original rows:", len(df))
print("New rows:", len(df_new))
print("Removed:", len(df) - len(df_new))


