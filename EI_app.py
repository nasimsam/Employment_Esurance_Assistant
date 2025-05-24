from langchain_openai import ChatOpenAI
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableMap
import chainlit as cl
from langchain_core.prompts import PromptTemplate
from bs4 import BeautifulSoup
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
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

    retriever = get_retriever()
    llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)
    classification_prompt = PromptTemplate.from_template(
    """Given Chat history and the context, determine the output.

    Context: {context}

    Chat history: {chat_history}

    input: {input}
    How to determine the output:

        Instructions:
        1. Analyse the 'context' and the 'user's input' to identify the specificity of the query and the scope of the information in 'context'.
        2. If there is a eligibilty condition in 'Context' that is not addressed in 'User Input' and 'Chat history' then follow up question is required, output '#'.
        3. If user ask about weekly benefits or rate in 'input' output '#'.
        4. If the user provides information regarding the following aspects: employment status, weekly salary and hourly rate or the 'input' includes either "Yes" or "No," then output "~".
        //EXAMPLES of output '~'//
        User: Yes. I am currently employed and worked more than 600 hours in the last 52 weeks
        //END OF EXAMPLES//

       
        //EXAMPLES of output '#'//
        User: I need to take care of my sick mother. Can I apply for EI?

        User: My husband wants to get an EI extended parental benefits, how does it work?
        //END OF EXAMPLES//
        
        
        output: '~' or '#'  """
    )
    follow_up_prompt = PromptTemplate.from_template(
        """
        User Input: {input}

        Chat history: {chat_history}

        Context: {context}
        Based on the information provided in 'context' and 'chat_history':\
        If there is missing information or there is a need for clarification, or there is a need for requirement verification, \
        then ask a follow up question to ensure that you have all the necessary information to respond effectively. \
        example of follow up questions:
        //EXAMPLES//
        User: What will be my weekly benefits or rate?

        Bot: Could you please provide your weekly hours or hourly rate?

        User: I need to take care of my sick mother. Can I apply for EI?

        Bot: Could you please provide more details about your mother's health condition? Additionally, do you have any medical documentation available? Lastly, is your mother currently residing outside the country?

        User: My husband wants to get an EI extended parental benefits, how does it work?

        Bot: Have you husband worked more than 600 hours in the last 52 weeks or since his last claim? Also, is he the biological or adoptive parent of the child?
        //END OF EXAMPLES//

    Follow-Up Question:"""
    )

    answer_prompt = PromptTemplate.from_template(
    """
    if hourly rate and weekly hours are provided use the following formula to calculate the weekly EI benefit rate 
    ((hourly rate * 20* weekly hour)/required number of best weeks ) * 55%
    = weekly EI benefit rate ( if more than $695 then return $695)
    where the required number of best weeks = 20

    Else, provide a comprehensive answer based on the user's input and the context.

    Chat history: {chat_history}

    Context: {context}

    Answer:"""
    )

    classification_chain = classification_prompt | llm
    conversation_history = [
        {"role": "system", "content": "Hi! I'm ready to help you with questions about Employment Insurance benefits"}]
    def get_conversation_history(user_input):
            conversation_history.append({"role": "user", "content": user_input})
            return conversation_history
    def route(output):
        classification = output['classification']
        #print(output["chat_history"])
        if classification.content == "#":
            return follow_up_prompt | llm
        else:
            return answer_prompt | llm

    # Create the chain
    rag_chain = (
        RunnableMap({
        "input": RunnablePassthrough(),
        "context": lambda x: retriever.get_relevant_documents(x["input"]),
    })
    | RunnableMap({
        "input": lambda x: x["input"],
        "context": lambda x: "\n".join([doc.page_content for doc in x["context"]]),
        "chat_history": lambda x: get_conversation_history(x["input"]),
    })
    | RunnableMap({
        "input": lambda x: x["input"],
        "context": lambda x: x["context"],
        "chat_history": lambda x: x["chat_history"],
        "classification": lambda x: classification_chain.invoke({"input": x["input"], "context": x["context"], "chat_history": x["chat_history"]}),
    })
    | RunnableLambda(route)
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
            {"input": message.content},
            config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
         ):
            await msg.stream_token(chunk.content)

    await msg.send()


if __name__ == "__main__":
    cl.run()
     