import json

with open("In-Context Learning and Prompt Templates-v3-GenAIcourse.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source = "".join(cell.get("source", []))
        if "%pip install" in source:
            new_source = "%pip install langchain langchain-community langchain-core --break-system-packages\n"
            cell["source"] = [new_source]

with open("In-Context Learning and Prompt Templates-v3-GenAIcourse.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
