import os



from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY") ## abhay


llm = ChatOpenAI(model = "gpt-3.5-turbo",api_key=api_key)
embeddings = OpenAIEmbeddings(model = "text-embedding-3-small",api_key=api_key)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
import uuid
config = {"configurable": {"thread_id": str(uuid.uuid4())}}

from langgraph.checkpoint.memory import MemorySaver

from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document
from langgraph.graph import StateGraph,START,END
from typing import List
from typing_extensions import TypedDict

tavily_api_key = os.getenv("TAVILY_API_KEY")
os.environ["TAVILY_API_KEY"] = tavily_api_key
tavily_search = TavilySearchResults(k=3)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
import warnings
warnings.filterwarnings("ignore")

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]
docs = [WebBaseLoader(url).load() for url in urls] ## i loaded text data from website to my local system
docs_list = [item for sublist in docs for item in sublist] ## i put all data from all websites in a single list
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 200,chunk_overlap = 10) ## created text splitter 
docs_split = text_splitter.split_documents(docs_list) ## splitted my text data into chunks
## now i will be creating the vector store where whole text data will be stored in form of word vector embeddings
vectorstore = Chroma.from_documents(
    documents=docs_split,
    embedding=embeddings,
    collection_name="rag-chrome"
)

retriever = vectorstore.as_retriever()
from chromadb import PersistentClient
from langchain_community.vectorstores import Chroma

# Initialize ChromaDB client with explicit tenant and database
client = PersistentClient(path="./chroma_db", tenant="default_tenant", database="default_database")

# Pass the client to your vector store
vectorstore = Chroma.from_documents(documents=docs_split, embedding=embeddings, client=client)
retriever = vectorstore.as_retriever()


prompt = hub.pull("rlm/rag-prompt")

rag_chain = prompt | llm | StrOutputParser() ## created the rag chain

class Grader(BaseModel):
    "binary_score for relevance check on the retriever documents"
    binary_score:str = Field(
        description="Documents are relevant to the question 'yes' or 'no' "
    )
llm_structured = llm.with_structured_output(Grader)
system = """You are a document grader. You will be given a question and documents.If the question is relevant to
the documents then you will return binary_score = 'yes' and if the question is not relevant to documents
then you will return binary_score = 'no'.check the question and document carefully then only make any decision  
             
         """
grade_prompt = ChatPromptTemplate.from_messages(
    [("system",system),
     ("human", "documents/n/n{documents} and question/n/n {question}")]

)
document_grader = grade_prompt| llm_structured

system = """  You are a question rewriter. You will be given a question. Your job is to understand the context of the question carefully
and update the question so that we can extract maximum information with the help of this question.Do not add any extra thing to questio.
if you are not able to do some addition on it .let the question as it as.
              
               """
prompt = ChatPromptTemplate.from_messages(
    [
        ("system",system),
        ("human","question:{question}")
    ]
)

question_rewriter = prompt | llm | StrOutputParser()


class State(TypedDict):
    question:str##
    documents:List[str]
    web_search:str
    response:str

class chatbot:##
    def __init__(self):
        pass
    def retrieve(self,state):
        "This function will retrieve the relevant documents from the vector database based on the user question"
        question = state["question"] 
        documents = retriever.invoke(question)

        return {"question":question,"documents":documents}
    def grade_documents(self,state):
        "This function will grade documents based on the question"
        question = state["question"] ## user question
        documents = state["documents"] ## documents fetched form the vector datavbase

        filtered_docs = []
        web_search = "no"

        for d in documents:
            score = document_grader.invoke({"question":question,"documents":d})
            binary_score = score.binary_score
            if binary_score == "yes":
                filtered_docs.append(d)
            else:
                web_search = "yes"
        return {"question":question,"documents":filtered_docs,"web_search":web_search}
    def generate(self,state):
        "This function will generate response based on the question and the retrieved documents, this will be used if web_search == 'no' "
        question = state["question"]
        documents = state["documents"]

        generated_response = rag_chain.invoke({"question":question,"context":documents})

        return {"question":question,"documents":documents,"response":generated_response}
    def transform_query(self,state):
        "This function will rewrite the query, so that this query may able to fetch maximum information from the web search"
        question = state["question"]
        documents = state["documents"]

        new_question = question_rewriter.invoke({"question":question})
        return {"question":new_question,"documents":documents}
    
    def net_search(self,state):
        question = state["question"]
        documents = state["documents"]
        response = tavily_search.invoke({"query":question})
        content = "/n".join([d["content"] for d in response])
        document = Document(page_content=content)
        documents.append(document)
        return {"question":question,"documents":documents}
    def decide_to_generate(self,state):
        "this function takes the decision either to generate or transform the query"
        question = state["question"]
        documents = state["documents"]
        web_search = state["web_search"]

        if web_search == "yes":
            return "transform_query" 
        else:
            return "generate"
    def __call__(self):
        memory = MemorySaver()
        workflow = StateGraph(State)
        workflow.add_node("retrieve",self.retrieve)
        workflow.add_node("document_grader",self.grade_documents)
        workflow.add_node("generate",self.generate)
        workflow.add_node("net_search",self.net_search)
        workflow.add_node("transform_query",self.transform_query)
        ## created all the nodes now i will be creating all the edges
        workflow.add_edge(START,"retrieve")
        workflow.add_edge("retrieve","document_grader")
        ## now using conditional edge
        workflow.add_conditional_edges("document_grader",self.decide_to_generate,{"transform_query":"transform_query","generate":"generate"})
        workflow.add_edge("transform_query","net_search")
        workflow.add_edge("net_search","generate")
        workflow.add_edge("generate",END)
        graph = workflow.compile(checkpointer=memory)
        self.app = graph
        return self.app
## till here i have written all the code for the chatbot
## now its time to execute the code for the chatbot
if __name__ == "__main__":
    mybot = chatbot()
    app = mybot()
    inputs = {"question":"what is memory in agent?"}
    response = app.invoke(inputs,config=config)
    print(response["response"])




            
 



