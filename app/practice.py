import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

print("Loaded key:", os.getenv("OPENAI_API_KEY"))

import chainlit as cl
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate  
from langchain.schema import StrOutputParser
from langchain.chains import LLMChain

@cl.on_chat_start
async def on_chat_start():
  model = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo")
  streaming = True
  prompt = ChatPromptTemplate.from_messages([
      ("system", "You are a chainlit GPT, helpful assistant."),
      ("human", "{question}")
  ])

  chain = LLMChain(
      llm=model,
      prompt=prompt,
      output_parser=StrOutputParser(),
      verbose=True,
  )

  cl.user_session.set("chain", chain)
# let's save the chain from user_session so we do not have to create it again
# every single time we receive a message 

@cl.on_message
async def main(message: cl.Message):
  chain = cl.user_session.get("chain")

  if not chain:
    await cl.Message("Chain not found, please start the chat first.").send()
    return
  # run the chain with the message content
  response = await chain.arun(
    
    question=message.content,
    callbacks=[cl.AsyncLangchainCallbackHandler()]
    )

  await cl.Message(message.content).send()