import pandas as pd
from sklearn.model_selection import train_test_split
import os

def clean_text(text):
    text = str(text).lower()
    text = text.replace("\n", " ")
    return text

# Load the dataset (semicolon delimiter!)
df = pd.read_csv("dataset.csv")

# Clean column names
df.columns = ["text", "label"]

# Clean text
df["text"] = df["text"].apply(clean_text)

# Create output directory
output_dir = "."
# No need to create folder since files are saved next to script

# Split dataset
train, temp = train_test_split(df, test_size=0.2, random_state=42)
val, test = train_test_split(temp, test_size=0.5, random_state=42)

# Save files
train.to_csv("data_train.csv", index=False)
val.to_csv("data_val.csv", index=False)
test.to_csv("data_test.csv", index=False)

print("Preprocessing done!")
print("Files created:")
print(" - data_train.csv")
print(" - data_val.csv")
print(" - data_test.csv")
