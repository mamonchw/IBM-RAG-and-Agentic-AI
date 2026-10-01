import json

with open("In-Context Learning and Prompt Templates-v3-GenAIcourse.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    source = "".join(cell.get("source", []))
    new_source = source
    if "GenParams().get_example_values()" in new_source:
        new_source = new_source.replace("GenParams().get_example_values()", "print('Ollama model uses local defaults.')")
        
    if "GenParams.MAX_NEW_TOKENS:" in new_source:
        new_source = new_source.replace("GenParams.MAX_NEW_TOKENS:", '"max_tokens":')
        new_source = new_source.replace("GenParams.TEMPERATURE:", '"temperature":')
        
    if "ibm/granite-4-h-small" in new_source:
        new_source = new_source.replace("ibm/granite-4-h-small", "llama3.1")
        new_source = new_source.replace("## Or you can use other LLMs available via watsonx.ai", "")
        new_source = new_source.replace('url = "https://us-south.ml.cloud.ibm.com"', "")
        new_source = new_source.replace('project_id = "skills-network"', "")
        
    if "To explore additional commonly used parameters, you can run the code `GenParams().get_example_values()`." in new_source:
        new_source = new_source.replace("To explore additional commonly used parameters, you can run the code `GenParams().get_example_values()`.", "Ollama handles parameters like temperature natively.")

    if new_source != source:
        cell["source"] = [s + "\n" if not s.endswith("\n") else s for s in new_source.split("\n")][:-1]
        
with open("In-Context Learning and Prompt Templates-v3-GenAIcourse.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
