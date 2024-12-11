import streamlit as st
#Agent and LLM
from langchain import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, Tool, ConversationalAgent
from langchain.memory import ConversationBufferMemory


#MEMORY
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory

#TOOLs
from langchain.utilities import GoogleSerperAPIWrapper

#ENV
import os
from dotenv import load_dotenv

load_dotenv("key.env")

#KEYS

SERPER_API_KEY = os.getenv("SERPER_API_KEY")

search = GoogleSerperAPIWrapper(serper_api_key=SERPER_API_KEY)


tools = [
    Tool(name="Seach",
    func=search.run,
    description="Usual when you need to search for news")
]

##Necessário escrever em português, porque o modelo estava respondendo em inglês. 
#Prefix
prefix = """"
Você é uma ferramenta de busca que ajuda o usuário a encontrar notícias interessantes e reais sobre a geopolítica mundial. Você tem acesso a uma ferramenta. Crie um textos de dois para três parágrafos sobre a geopolítica mundial.
"""

suffix = """
Chat History: {chat_history}
Last Question: {input}
"""

prompt = ConversationalAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input","chat_history"],
    )

#Memory

msg = StreamlitChatMessageHistory()

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(messages=msg,
                                                       memory_key="chat_history",
                                                         return_messages=True)
memory = st.session_state.memory

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
#Agent

llm_chain = LLMChain(llm=llm,
                      prompt=prompt,
                      verbose=True)

agent = ConversationalAgent(llm_chain=llm_chain,
                                memory=memory,
                                verbose=True,
                                max_interactions=5,
                                tools=tools)

#AgentExecutor
executor = AgentExecutor.from_agent_and_tools(agent=agent,
                         tools=tools,
                         memory=memory,
                         verbose=True)


query = st.text_input("O que você quer saber sobre a geopolítica global?")

if query:
    with st.spinner("Estou procurando a informação..."):
        response = executor.run(query)
        st.info(response,icon="🔍")




