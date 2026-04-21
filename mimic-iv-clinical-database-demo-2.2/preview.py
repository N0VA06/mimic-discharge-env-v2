import os
import pandas as pd

base_path = "."  # current directory
output_file = "preview.txt"

with open(output_file, "w") as f:
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                try:
                    df = pd.read_csv(file_path, nrows=5)
                    
                    header = f"\n===== {file_path} =====\n"
                    print(header)
                    print(df)

                    f.write(header)
                    f.write(df.to_string(index=False))
                    f.write("\n")

                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
