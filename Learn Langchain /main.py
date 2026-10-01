import os
from dotenv import load_dotenv

load_dotenv()

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

# 1. Define the Prompt Template with a variable ({email_content})
prompt = PromptTemplate(
    # input_variables=["email_content"],
    template="The wind is"
)

# 2. Initialize the language model
llm = ChatOllama(model="llama3.1", temperature=0)

# 3. Initialize an output parser to clean up the string output
output_parser = StrOutputParser()

# 4. Chain the components together using the LCEL pipe operator (|)
classification_chain = prompt | llm | output_parser

# 5. Invoke the chain with runtime data
email_text = "My account is completely locked and I can't access my billing data!"
result = classification_chain.invoke()

print(result)
