import json

with open("In-Context Learning and Prompt Templates-v3-GenAIcourse.ipynb", "r") as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "markdown":
        source = "".join(cell.get("source", []))
        if "Watsonx" in source or "watsonx" in source or "API Disclaimer" in source:
            print(f"Cell {i} contains relevant markdown.")
            print(source)
            print("-" * 40)
