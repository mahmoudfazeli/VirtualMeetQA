from langchain.prompts import PromptTemplate

# Modified prompt template for Q&A over workshops
prompt_template = """The following is a conversation between you and a workshop participant. 
As a workshop facilitator, you need to answer questions about the workshop topic.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
{context}

Question: {question}
Summarise the workshop in the 5 key takeaways in bullet points format in details.
-------
"""

# Create a PromptTemplate object with the modified template
prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# Modified query template for Q&A over workshops
query_template = """Given this workshop topic:{workshop_topic}, please summarize the workshop in 5 key takeaways in bullet points format in details.
Then, ask the ask the participant of the workshop if they have any specifiy question about one of the takeaways.
"""