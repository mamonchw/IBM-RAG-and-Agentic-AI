import json

with open("In-Context Learning and Prompt Templates-v3-GenAIcourse.ipynb", "r") as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        source = "".join(cell.get("source", []))
        if "WatsonxLLM" in source or "ibm_watsonx_ai" in source or "def llm_model" in source:
            print(f"Cell {i} contains relevant code.")
            print(source)
            print("-" * 40)
