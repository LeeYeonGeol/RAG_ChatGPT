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


## save db
df = pd.read_csv("./hitter_record_1982_to_2023_cleaned(ver.2).csv")
documents = []
'''
for name in df['이름'].unique():
    document = f"{name} 선수에 관한 문서입니다. \n"
    for idx, row in df[df['이름'] == name].iterrows():
        document += str(row.to_dict())[1:-1]
        document += '\n'

    documents.append(Document(page_content=document))
'''
#embeddings = CohereEmbeddings(model="embed-multilingual-light-v3.0", cohere_api_key=config.cohere_api_key)

for name in df['이름'].unique():
    document = f"{name} 선수에 관한 문서입니다. 시즌별 기록을 확인 할 수 있습니다.\n"
    metadata = defaultdict(dict)
    for idx, row in df[df['이름'] == name].iterrows():
        metadata[str(row['시즌']) + ' 시즌'] = row.to_dict()
    '''
    for idx, row in df[df['이름'] == name].iterrows():
        document += str(row.to_dict())[1:-1]
        document += '\n'
    '''
    documents.append(Document(page_content=document, metadata=metadata))

embeddings = OpenAIEmbeddings(openai_api_key = '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')

db = Qdrant.from_documents(
    documents,
    embeddings,
    path="/tmp/qdrant_last",
    collection_name="my_documents",
    force_recrete=True
    )

## load llm
project_id = 'kboproject'
creds, _ = google.auth.default(quota_project_id=project_id)
vertexai.init(project=project_id, credentials=creds)

llm = VertexAI(model_name="text-bison", max_output_tokens=1024, temperature=0.3)

## hitter chain
# 템플릿
template = '''아래 글만을 참고해서 질문에 대한 답을 해줘.
            {context}

            질문: {question}

            답변:
'''
retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 1})
# 프롬프트 템플릿 생성
prompt = PromptTemplate(template=template, input_variables=["question", "context"])
# hitter chain 객체 생성
hitter_detail_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

## wiki chain
retriever = WikipediaRetriever(lang='ko', search_type="mmr")
# wiki chain 객체 생성
wiki_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | VertexAI(model_name="text-bison", max_output_tokens=1024, temperature=0.9)
    | StrOutputParser()
)

## agent 생성
tools = [
    Tool(
        name="타자 시즌별 기록",
        func=hitter_detail_chain.invoke,
        description="해당 타자의 시즌별 기록을 알아볼 때 유용합니다.",
    ),

    Tool.from_function(
        name="위키피디아",
        func= wiki_chain.invoke,
        description="해당 타자의 전반적인 정보를 알고 싶을 때 유용합니다."
    )
]

agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)


