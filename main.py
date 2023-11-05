from langchain.llms import OpenAI
from langchain.chains import RetrievalQA, ConversationChain
from langchain.memory import ConversationBufferMemory
from backend.conf import ChainConf
from backend.embedding import load_retriever
from backend.prompt import prompt, query_template

chain_conf = ChainConf()
llm = OpenAI(
    model=chain_conf.llm_model,
    temperature=0.2,
    top_p=0.9,
)
conv_chain = ConversationChain(
    memory=ConversationBufferMemory(),
    llm=llm
)

def initialize_workshop(workshop_topic: str):
    if not isinstance(workshop_topic, str) or not workshop_topic:
        raise ValueError("Invalid workshop_topic value")

    chain_type_kwargs = {"prompt": prompt}
    retriever = load_retriever(chain_conf)
    
    # Initialize RetrievalQA
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs=chain_type_kwargs,
    )

    # Use RetrievalQA to get the answer
    query = query_template.format(workshop_topic=workshop_topic)
    qa_response = qa_chain.run({"query": query})
    # Store the QA response in ConversationChain's memory
    conv_chain.memory.save_context({"query": "query"}, {"output": qa_response})

    return qa_response

def get_response(user_input: str):
    conv_response = conv_chain.run({"input": user_input})
    return conv_response

def main():
    print("Welcome! Please provide the workshop topic:")
    workshop_topic = input("Workshop Topic: ")
    #workshop_topic = "Getting into the User's problem"
    initial_response = initialize_workshop(workshop_topic)
    print(initial_response)
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        conv_response = get_response(user_input)
        print(f"AI: {conv_response}")

if __name__ == "__main__":
    main()
