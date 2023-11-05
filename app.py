import chainlit as cl
from backend.conf import ChainConf
from main import initialize_workshop, get_response

@cl.on_chat_start
async def start_session():
    # Load configurations and store in the user session for later use
    chain_conf = ChainConf()
    cl.user_session.set("chain_conf", chain_conf)
    
    # Prompt the user to provide the workshop topic
    await cl.Message(content="Welcome! Please provide the workshop topic:").send()

@cl.on_message
async def workshop_qa(message: cl.Message):
    chain_conf = cl.user_session.get("chain_conf")
    
    # If it's the first message, treat it as the workshop topic
    if not cl.user_session.get("workshop_initialized"):
        initial_response = initialize_workshop(
            #chain_conf,
            message.content
            )
        cl.user_session.set("workshop_initialized", True)
        await cl.Message(content=initial_response).send()
    else:
        # Get the response based on the user's message
        response = get_response(message.content)
        await cl.Message(content=response).send()
