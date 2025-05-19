from langchain_openai import ChatOpenAI
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
import chainlit as cl
from langchain_core.prompts import PromptTemplate
from bs4 import BeautifulSoup
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain.memory import ConversationBufferMemory
import os

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_retriever():
    # Load
    url = "https://www.canada.ca/en/employment-social-development/programs/ei/ei-list/reports/digest.html"
    loader = RecursiveUrlLoader(
        url=url,
        max_depth=50,
        extractor=lambda x: BeautifulSoup(x, "html.parser").text
    )
    docs = loader.load()

    # Split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)

    # Embed
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

    # Return retriever with search kwargs
    return vectorstore.as_retriever(search_kwargs={"k": 3})

@cl.on_chat_start
async def start():

    # Initialize the chain components
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    retriever = get_retriever()
    llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)
    clarifying_guidelines="""Given the 'user’s question': {question}

    And the detailed information provided in 'context' and 'chat_history': {context} and {chat_history},

    Determine, whether a clarifying question is required.

    Instructions:
    1. Analyse the 'context' and the 'user's question' to identify the specificity of the query and the scope of the information in 'context'.
    2. If the query aligns well with a specific part of the 'context' and 'chat_history' that provides a comprehensive answer, output '~'.
    3. If the query is broad and the 'context' include an eligibility condition that did not addressed in 'chat_history', you should ask for the eligibility requirements., output '#' and guide the chatbot to ask for clarification.
    

    Output format: [Decision: '~' or '#', (if '#') then clarification is required.]"""
    template = """Given the guidelines provided in 'clarifying question check' as
    {clarifyingQuestionCheck},
    the user's question stated in
    {question},
    and the information in both 'context' and 'chat_history' as
    {context} and {chat_history},
    If clarification is needed, ask a follow-up question to gather more information. The question should be specific and relevant to the user's query.

    //EXAMPLES//
    User: I need to take care of my sick mother. Can I apply for EI?

    Bot: Could you please provide more details about your mother's health condition? Additionally, do you have any medical documentation available? Lastly, is your mother currently residing outside the country?

    User: My husband wants to get an EI extended parental benefits, how does it work?

    Bot: Have you husband worked more than 600 hours in the last 52 weeks or since his last claim? Also, is he the biological or adoptive parent of the child?

    //END OF EXAMPLES//

    Output the question clearly and concisely, with no additional text.
    else Output answer the question based on the information in 'context' and 'chat_history' as
    {context} and {chat_history}."""
    prompt = PromptTemplate(input_variables=["chat_history", "input"],
        template=template)
    
    # Create the chain
    rag_chain = (
        {
            "clarifyingQuestionCheck": lambda x: clarifying_guidelines,
            "context": lambda x: format_docs(retriever.invoke(x["question"])),
            "question": RunnablePassthrough(),
            "chat_history": lambda x: memory.load_memory_variables({})["chat_history"]
        }
        | prompt
        | llm
    )

    
    # Store the chain in the user session

    
    cl.user_session.set("chain", rag_chain)



    # Send an initial message
    await cl.Message(content="Hi! I'm ready to help you with questions about Employment Insurance benefits").send()

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")  # Get the chain from the session

    if chain is None:
        await cl.Message(content="Error: Chain not initialized. Please restart the chat.").send()
        return

    msg = cl.Message(content="")

    async with cl.Step("Fetched Content 🤔"):
        async for chunk in chain.astream(
            {"question": message.content},
            config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
        ):
            await msg.stream_token(chunk.content)

    await msg.send()

if __name__ == "__main__":
    cl.run()
     