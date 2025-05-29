import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

import chainlit as cl
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import ChatPromptTemplate  
from langchain.schema import StrOutputParser
from langchain.chains import LLMChain
from transformers import AutoTokenizer, pipeline

@cl.on_chat_start
async def on_chat_start():
    model_name = os.getenv("HF_MODEL_NAME", "gpt2")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Detect if it's a T5 model
        if "t5" in model_name or "flan" in model_name:
            from transformers import T5ForConditionalGeneration
            model = T5ForConditionalGeneration.from_pretrained(model_name)
            pipe = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=512,
                temperature=0.3,
                top_p=0.95,
                repetition_penalty=1.15
            )
        else:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(model_name)
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=512,
                temperature=0.3,
                top_p=0.95,
                repetition_penalty=1.15
            )
        llm = HuggingFacePipeline(pipeline=pipe)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            ("human", "{question}")
        ])
        chain = LLMChain(
            llm=llm,
            prompt=prompt,
            output_parser=StrOutputParser(),
            verbose=True,
        )
        cl.user_session.set("chain", chain)
        await cl.Message(content="Model loaded successfully! You can start chatting.").send()
    except Exception as e:
        await cl.Message(content=f"Error loading model: {str(e)}").send()
        return

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    if not chain:
        await cl.Message(content="Chain not found, please start the chat first.").send()
        return
    try:
        response = await chain.arun(
            question=message.content,
            callbacks=[cl.AsyncLangchainCallbackHandler()]
        )
        await cl.Message(content=response).send()
    except Exception as e:
        await cl.Message(content=f"Error generating response: {str(e)}").send() 