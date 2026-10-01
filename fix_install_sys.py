import json

with open("In-Context Learning and Prompt Templates-v3-GenAIcourse.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source = "".join(cell.get("source", []))
        if "pip install langchain" in source:
            new_source = "import sys\n!{sys.executable} -m pip install langchain langchain-community langchain-core --break-system-packages\n"
            cell["source"] = [s + "\n" if not s.endswith("\n") else s for s in new_source.split("\n")][:-1]

with open("In-Context Learning and Prompt Templates-v3-GenAIcourse.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
