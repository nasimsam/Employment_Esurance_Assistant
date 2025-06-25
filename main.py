from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableMap
from langchain.embeddings import OpenAIEmbeddings
import os
from dotenv import load_dotenv
from bs4 import BeautifulSoup as Soup
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_core.prompts import ChatPromptTemplate
import operator
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict, Annotated
from langchain_core.messages import HumanMessage, SystemMessage, AnyMessage, get_buffer_string
from langchain.embeddings import OpenAIEmbeddings
from langchain.tools import Tool
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.graph import StateGraph, END
from langchain.schema import Document
from pydantic import BaseModel, Field
from langsmith import traceable

os.environ["LANGCHAIN_TRACING_V2"] = "true"
# define your langchain project name here
os.environ["LANGCHAIN_PROJECT"] = "EI Assistant"

load_dotenv()
OPENAI_API = os.getenv('OPENAI_API_KEY')
LANGCHAIN_API = os.getenv('LANGCHAIN_API_KEY')
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)


# Load
url = "https://www.canada.ca/en/employment-social-development/programs/ei/ei-list/reports/digest.html"
loader = RecursiveUrlLoader(
    url=url, max_depth=50, extractor=lambda x: Soup(x, "html.parser").text
)
docs = loader.load()

# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
splits = text_splitter.split_documents(docs)

vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(api_key=OPENAI_API))
retriever = vectorstore.as_retriever()

system_prompt = (
    """Given User Input, Chat history and the context, Only output '#' or '~' base of the following roles:
User Input: {input}

Context: {context}

Chat history: {chat_history}


Instructions:
1. Analyze the 'Context', 'Chat history', and the 'User Input' to identify the specificity of the query and the scope of the information in 'Context'.

2. Output '#' for the following cases:
    - If there is missing information in 'Chat history' or 'User Input'
    - If the user question include 'Can' or 'Could' and there is a need for clarification or requirement verification.
    - If user question about his/her EI options, and there is need to requirement verification that is not covered in the 'Chat_hisory' or 'User Input'.
   //EXAMPLES of output '#'//
   User: I need to take care of my sick mother. Can I apply for EI?
   User: My husband wants to get EI extended parental benefits, how he can apply for it?
   User: I am currently receiving regular EI benefits, however I am planing to travel abroad, can I still receive my benefits?
   //END OF EXAMPLES//

3. Output '~' for the following cases:
   - Look in 'Chat history' if the most recent message with 'AIMessage' tag includes a question.
   - The user asks a general question.
   - The user mentions that he/she can provide proof or documentation.
   - If the user provides any information regarding his/her eligibility, output '~'.
   - Provides information regarding the following aspects: employment status, weekly salary, and hourly rate.
   - 'User Input' includes either "Yes" or "No".
   //EXAMPLES of output '~'//
   User Input: What types of EI benefits do we have?
   User Input: My mother is critically ill. I am looking for family care benefits.
   User Input: I am interested in a general overview of all types of EI benefits.
   User Input: Yes. I am currently employed and worked more than 600 hours in the last 52 weeks.
   //END OF EXAMPLES//
 """
  
)

classification_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

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

        User: I am currently receiving regular EI benefits, however I am planning to travel abroad, can I still receive my benefits?

        Bot: Could you please share the purpose of your upcoming travel abroad? Additionally, what is the expected duration of your trip?
        //END OF EXAMPLES//

    Follow-Up Question:"""
    )

answer_prompt = PromptTemplate.from_template(
    """
    Use the retrieved 'Context' and 'Chat history' to answer the question. If you don't know the answer, just say that you don't know. Provide an answer in one pharagraph.

    Chat history: {chat_history}

    Context: {context}

    Answer:"""
)


def route(output):
    classification = output['classification']
    if classification == "#":
        return follow_up_prompt
    else:
        return answer_prompt
    
rag_chain = (
    RunnableMap({
        "input": lambda x: x["input"],
        "context": lambda x: x["context"],
        "chat_history": lambda x: x["chat_history"],
        "classification": lambda x: x["classification"],
    })| route
  )   


def Calculate_EI_benefit(hourly_rate: float, weekly_hours: float) -> float:
    """
    Calculate the estimated weekly EI benefit.

    :param hourly_rate: User's hourly wage
    :param weekly_hours: Number of hours worked per week
    :param family_income: Net family income (optional)
    :param has_children: Whether the user has children (optional)
    :return: Estimated weekly EI benefit
    """
    weekly_earnings = hourly_rate * weekly_hours
    benefit_rate = 0.55

    # Check for Family Supplement eligibility

    weekly_benefit = weekly_earnings * benefit_rate
    return min(weekly_benefit, 695.00)
    
tools = [Calculate_EI_benefit]
llm_with_tools = llm.bind_tools(tools) 



class CitedAnswer(BaseModel):
    answer: str = Field(..., description="The answer to the user question.")
    citations: List[int] = Field(..., description="List of source document indices used.")


structured_llm = llm.with_structured_output(CitedAnswer)
checkpointer = MemorySaver()
sys_msg = SystemMessage(content="You are a helpful Employment Assistant (EI) assistant only if user ask for benefit rate or to calculate EI benefit use tool to calculate, ask follow up questions if there is missing information or clarity is needed")
# Define the state schema
class GraphState(TypedDict):
    user_input: str
    messages: Annotated[List[AnyMessage], operator.add]
    documents: List[Document]
    citation : str

# System message


# Node
def assistant(state: GraphState):
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# Define the calculator tool

"""
retrieve_documents
- Returns documents fetched from a vectorstore based on the user's question
"""
@traceable(run_type="chain")
def retrieve_documents(state: GraphState):
    messages = state.get("messages", [])
    user_input = state["user_input"]
    documents = retriever.invoke(f"{get_buffer_string(messages)} {user_input}")
    return {"documents": documents}


@traceable(run_type="chain")
def generate_response(state: GraphState):
    user_input = state["user_input"]
    messages = state["messages"]
    documents = state["documents"]
    formatted_docs = "\n\n".join(doc.page_content for doc in documents)
    classification = classification_prompt.invoke({"input": user_input, "chat_history": messages , "context": formatted_docs})
    classification_response = structured_llm.invoke(classification)
    print
    response = rag_chain.invoke({"input": user_input, "chat_history": messages , "context": formatted_docs, "classification": classification_response.answer})
    generation = llm.invoke(response)
    return {"documents": documents, "messages": [generation], "citation": classification_response.citations}


class EIAgent:
    def __init__(self):
        self.workflow = StateGraph(GraphState)
        self.workflow.add_node("assistant", assistant)
        self.workflow.add_node("tools", ToolNode(tools))
        self.workflow.add_node("retrieve_documents", retrieve_documents)
        self.workflow.add_node("generate_response", generate_response)
        self.workflow.add_edge(START, "assistant")
        self.workflow.add_conditional_edges(
            "assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
    )
        self.workflow.add_edge("assistant", "retrieve_documents")
        self.workflow.add_edge("tools", "assistant")
        self.workflow.add_edge("retrieve_documents", "generate_response")
        self.workflow.add_edge("generate_response", END)
        self.app = self.workflow.compile(checkpointer=checkpointer)

    def ask(self, question: str, config) -> str:
        
        result = self.app.invoke({"messages": [HumanMessage(content=question)],"user_input" : question}, config=config)
        citation = result.get("citation", [])
        return {"answer": result.get("messages", []), "with citations": citation}




    


