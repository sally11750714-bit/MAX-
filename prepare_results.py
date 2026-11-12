
import json
from pathlib import Path

root = Path(r"C:\MAX")
source = root / "outputs"
destination = root / "results"

categories = ["carpet", "bottle", "hazelnut"]

def clean_paths(value):
    if isinstance(value, dict):
        return {
            k: clean_paths(v)
            for k, v in value.items()
        }

    if isinstance(value, list):
        return [clean_paths(v) for v in value]

    if isinstance(value, str):
        path = value.replace("\\", "/")
        prefix = str(root).replace("\\", "/") + "/"

        if path.lower().startswith(prefix.lower()):
            return path[len(prefix):]

    return value

for category in categories:
    output = destination / category
    output.mkdir(parents=True, exist_ok=True)

    for filename in ["metrics.json", "split_manifest.json"]:
        input_file = source / category / filename

        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        data = clean_paths(data)

        with open(output / filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"Saved: results/{category}/{filename}")

print("All results prepared successfully!")
