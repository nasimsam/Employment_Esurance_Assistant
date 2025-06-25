
import chainlit as cl
from chainlit.message import Message
import uuid
from main import EIAgent
agent = EIAgent()
@cl.on_chat_start
async def on_chat_start():
    thread_id = str(uuid.uuid4())
    cl.user_session.set("thread_id", thread_id)
    """Send a welcome message when the chat starts."""
    welcome_text = "Hi! I'm an AI-powered EI assistant and ready to help you with questions about Employment Insurance benefits. Please note that while I strive to provide accurate information, my responses may not always be correct or complete. Always verify critical information independently."
    await cl.Message(content=welcome_text).send()


@cl.on_message
async def on_message(message: Message):
    thread_id = cl.user_session.get("thread_id") # Unique ID for the session
    config = {"configurable": {"thread_id": thread_id}}
    """Handle incoming messages."""
    msg = cl.Message(content="")
    async with cl.Step("Fetched Content 🤔"):
        response= agent.ask(message.content, config)
        await msg.stream_token(response["answer"][-1].content + "\n\n")
        if str(response["with citations"]) != "" and str(response["with citations"])!= '[0]' and str(response["with citations"]) != "[]":
            await msg.stream_token("You can find the related information in the following chapters from [Digest of Benefit Entitlement Principles](https://www.canada.ca/en/employment-social-development/programs/ei/ei-list/reports/digest.html): "+ str(response["with citations"]).strip("[]"))
 
    await msg.send()

if __name__ == "__main__":
    cl.run()