from langchain.agents.openai_functions_agent.agent_token_buffer_memory import (
    AgentTokenBufferMemory,
)
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
#from langchain.callbacks.streaming_stdout_final_only import (
 #   FinalStreamingStdOutCallbackHandler,
#)
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.schema.messages import SystemMessage
from langchain.prompts import PromptTemplate, MessagesPlaceholder
from backend.embedding import load_retriever
from langchain.agents import AgentExecutor
import streamlit as st
from dotenv import load_dotenv
from backend.conf import ChainConf

load_dotenv()
chain_conf = ChainConf()


llm = ChatOpenAI(
    model=chain_conf.chat_model,
    temperature=0.2,
    #top_p=0.9,
    #streaming=True,
    #callbacks=[FinalStreamingStdOutCallbackHandler()]
)

msgs = StreamlitChatMessageHistory(key="special_app_key")
memory_key = "history"
memory = AgentTokenBufferMemory(memory_key=memory_key, llm=llm, chat_memory = msgs)
#memory = ConversationBufferMemory(memory_key=memory_key,return_messages=True, chat_memory=msgs)
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")


retriever = load_retriever(chain_conf)

tool = create_retriever_tool(
    retriever,
    "search_documents",
    "Searches and returns relevant documents.",
)
tools = [tool]

system_message = SystemMessage(
    content=(
        "You are a chatbot equipped to answer questions based on a variety of documents. "
        "These documents may be in any language, but your responses should always be in English. "
        "Unless explicitly mentioned otherwise, assume questions are related to the content of these documents. "
        "Do not attempt to answer irrelevant questions or those outside the scope of the documents. "
        "If unsure or the question is irrelevant, simply state that you do not know and request the user for further explanation."
        "Refrain from long responses. If a long explanation is needed, provide a brief summary in one paragraph "
        "and key points in bullet form."
    )
)

agent_prompt = OpenAIFunctionsAgent.create_prompt(
    system_message=system_message,
    extra_prompt_messages=[MessagesPlaceholder(variable_name=memory_key)],
)

agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=agent_prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    return_intermediate_steps=True,
)

for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)


if user_input := st.chat_input():
    st.chat_message("human").write(user_input)

    # As usual, new messages are added to StreamlitChatMessageHistory when the Chain is called.
    #response = llm_chain.run(prompt)
    response_data = agent_executor({"input": user_input}, return_only_outputs = True)
    response_output = response_data.get("output", "Sorry, I couldn't process your request.")
    st.chat_message("ai").write(response_output)