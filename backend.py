import config
import uvicorn
from fastapi import FastAPI

import vertexai
import google.auth
from langchain.llms import VertexAI
from langchain.chat_models import ChatVertexAI
from langserve import add_routes

from server import agent as rag_agent
from server import hitter_detail_chain
from server import wiki_chain
from server import llm

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)

add_routes( 
    app,
    hitter_detail_chain,
    path="/hitter_detail_chain"
)

add_routes( 
    app,
    rag_agent,
    path="/rag_agent"
)

add_routes( 
    app,
    wiki_chain,
    path="/wiki_chain"
)

add_routes( 
    app,
    llm,
    path="/llm"
)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
