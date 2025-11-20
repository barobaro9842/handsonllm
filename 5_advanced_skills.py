##langchain

from langchain_core.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
import os

load_dotenv()

open_ai_key=os.getenv("OPENAI_API_KEY")

llm = LlamaCpp(
    model_path="model/Phi-3-mini-4k-instruct-q4.gguf",
    n_gpu_layers=-1,
    max_tokens=500,
    n_ctx=4096,
    seed=42,
    verbose=False
)

chat_model = ChatOpenAI(openai_api_key=open_ai_key)
template = """
<|user|>
{input_prompt}<|end|>
<|assistant|>
"""
prompt = PromptTemplate(
    template=template,
    input_variables=["input_prompt"]
)

basic_chain = prompt | llm


# 템플릿 체인
from langchain_core.runnables import RunnableSequence, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

template = """
<|user|>
Create a title for a story about {summary}. Only return the title. <|end|>
<|assistant|>
"""

title_prompt = PromptTemplate(template=template, input_variables=["summary"])
output_parser = StrOutputParser()

title = RunnableSequence(
    {"summary": RunnablePassthrough()}
    | title_prompt
    | llm 
    | output_parser,
    name="title"
)

title.invoke("a girl that lost her mother")

# 체인 만들기
template = """
<|user|>
Describe the main character of a story about {summary} with the title {title}. Use only two sentences.
<|end|>
<|assistant|>
"""
character_prompt = PromptTemplate(
    template=template, input_variables=["summary", "title"]
)

character = RunnableSequence(
    {
        "summary": RunnablePassthrough(),
        "title": RunnablePassthrough()
    }
    | character_prompt
    | llm
    | output_parser,
    name="character"
)

template = """
<|user|>
Create a story about {summary} with the title {title}. The main character is: {character}. Only return the story and it cannot be longer than one paragraph.
<|end|>
<|assistant|>
"""
story_prompt = PromptTemplate(
    template=template,
    input_variables=["summary", "title", "character"]
)

story = RunnableSequence(
    {
        "summary": RunnablePassthrough(),
        "title": RunnablePassthrough(),
        "character": RunnablePassthrough()
    }
    | story_prompt
    | llm
    | output_parser,
    name="story"
)

llm_chain = title | character | story
llm_chain.invoke("a girl that lost her mother")



### Memory
# 메모리 X
basic_chain.invoke(
    {
        "input_prompt": "Hi! My name is Maarten. What is 1 + 1?"
    }
)

basic_chain.invoke(
    {
        "input_prompt": "What is my name?"
    }
)

# 메모리 O
template = """<|user|>Current conversation:{chat_history}

{input_prompt}<|end|>
<|assistant|>
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["input_prompt", "chat_history"]
)

from langchain_classic.memory import ConversationBufferMemory

memory = ConversationBufferMemory(memory_key="chat_history")