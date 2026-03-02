import os

root_dir = "./output"

for root, dirs, files in os.walk(root_dir):
    if "checkpoint.pkl" in files:
        file_path = os.path.join(root, "checkpoint.pkl")
        print(f"Deleting: {file_path}")
        os.remove(file_path)

print("Done.")