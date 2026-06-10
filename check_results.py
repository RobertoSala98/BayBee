import os
import json

base_dir = "output/ligen_2.0_ridge"

mapr_values = []
global_time_values = []

for folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder)
    
    # Ensure it's one of the abc_run* directories
    if os.path.isdir(folder_path) and folder.startswith("abc_run"):
        summary_path = os.path.join(folder_path, "summary.json")
        
        if os.path.exists(summary_path):
            with open(summary_path, "r") as f:
                data = json.load(f)
                
                # Adjust if MAPR is nested (see note below)
                if "MAPR" in data:
                    mapr_values.append(data["MAPR"])

                if "Global time" in data:
                    global_time_values.append(data["Global time"])

if mapr_values:
    avg_mapr = sum(mapr_values) / len(mapr_values)
    print(f"Average MAPR: {avg_mapr}")
else:
    print("No MAPR values found.")

if global_time_values:
    avg_global_time = sum(global_time_values) / len(global_time_values)
    print(f"Average Global time: {avg_global_time}")
else:    
    print("No Global time values found.")
