import config
import pandas as pd
from collections import defaultdict

import vertexai
import google.auth
from langchain.llms import VertexAI
from langchain.chat_models import ChatVertexAI
from langchain.schema import Document
from langchain.embeddings import CohereEmbeddings
from langchain.vectorstores import Qdrant
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers import WikipediaRetriever
from langchain.agents import Tool, AgentType, initialize_agent
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import SystemMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableMap
from langserve import RemoteRunnable
import google.auth
import gradio as gr


llm = RemoteRunnable("http://127.0.0.1:8000/llm/")

chain = RemoteRunnable("http://localhost:8000/hitter_detail_chain/")
wiki_chain = RemoteRunnable("http://localhost:8000/wiki_chain/")
rag_agent = RemoteRunnable("http://localhost:8000/rag_agent/")
'''
def predict(message, history):
    history_langchain_format = history + [[message, ""]]

    messages = "".join(["".join(["\n<human>:"+item[0], "\n<bot>:"+item[1]]) for item in history_langchain_format])

    response = chain.invoke(messages)

    return response.content
'''

## load llm
project_id = 'kboproject'
creds, _ = google.auth.default(quota_project_id=project_id)
vertexai.init(project=project_id, credentials=creds)

llm = VertexAI(model_name="text-bison", max_output_tokens=1024, temperature=0.3)

def predict(message, history):
    history_langchain_format = []
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))
    history_langchain_format.append(HumanMessage(content=message))
    gpt_response = llm(history_langchain_format)
    return gpt_response.content

gr.ChatInterface(predict).launch()
