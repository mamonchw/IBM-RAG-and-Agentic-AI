import json

with open("In-Context Learning and Prompt Templates-v3-GenAIcourse.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") == "code":
        source = "".join(cell.get("source", []))
        if "pip install" in source:
            print(f"Cell {i}:")
            print(source)
