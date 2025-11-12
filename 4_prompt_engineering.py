import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# 모델, 토크나이저 로드
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

# 파이프라인
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    max_new_tokens=500,
    do_sample=False,
)

# 프롬프트
messages = [
    {"role": "user", "content": "Create a funny joke about chickens"}
]

output = pipe(messages)
print(output[0]["generated_text"])

# 프롬프트 템플릿 적용
prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False)
print(prompt)

# temperature 조절 (확률이 낮은 답변을 할 확률을 높여줌)
output = pipe(messages, do_sample=True, temperature=1)
print(output[0]["generated_text"])

# top_p(누적 확률로 선택할 토큰의 개수 조절)
output = pipe(messages, do_sample=True, top_p=1)
print(output[0]["generated_text"])


# 프롬프트 구성요소
persona = "You are an expert in Large Language models. You excel at breaking down complex papers into digestible summaries. \n"
instruction = "Summarize the key findings of the paper provided. \n"
context = "Your summary should extract the most crucial points that can help researchers quickly understand the most vital information of the paper. \n"
data_format = "Create a bullet-point summary that outlines the method. Follow this up with a concise paragraph that encapsulates the main results. \n"
audience = "The summary is designed for busy researchers that quickly need to grasp the newest trends in Large Language Models. \n"
tone = "The tone should be professional and clear. \n"
text = "MY TEXT TO SUMMARIZE"
data = f"Text to summarize: {text}"

query = persona + instruction + context + data_format + audience + tone + data

########## few shot learning
# 모델에게 질문을 바로 하는 것이 아니라, 질문-답변 쌍을 예시로 제시하고 질문을 하는 것
one_shot_prompt = [
    {
        "role": "user",
        "content": "A 'Gigamaru' is a Japanese musical instrument. An example of a sentence that uses the word Gigamaru is: "
    },
    {
        "role": "assistant",
        "content": "I have a Ggigamaru that my uncle gave me as a gift. I love to play it at home."
    },
    {
        "role": "user",
        "content": "To 'screeg' something is to swing a sword at it. An example of a sentence that uses the word screeg is: "
    }
]

print(tokenizer.apply_chat_template(one_shot_prompt, tokenize=False))
outputs = pipe(one_shot_prompt)
print(outputs[0]["generated_text"])

######## Prompt chain 
# 전체 문제를 작은 문제로 쪼개기

# 1st chain
product_prompt = [
    {
        "role": "user",
        "content": "Create a name and slogan for a chatbot that leverages LLMs."
    }
]

outputs = pipe(product_prompt)
product_description = outputs[0]["generated_text"]
print(product_description)

# 2nd chain
sales_prompt = [
    {
        "role": "user",
        "content": f"Generate a very short sales pitch for the following product: {product_description}"
    }
]

outputs = pipe(sales_prompt)
sales_pitch = outputs[0]["generated_text"]
print(sales_pitch)


##### CoT (Chain of Thought)
# 추론을 포함한 One shot prompt
cot_prompt = [
    {
        "role": "uesr",
        "content": "Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can gas 3 tennis balls. How many tennis balls does he have now?"
    },
    {
        "role": "assistant",
        "content": "Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5 + 6 = 11. the answer is 11."
    },
    {
        "role": "user",
        "content": "The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?"
    }
]

outputs = pipe(cot_prompt)
print(outputs[0]["generated_text"])

# zero shot CoT (LTSS)
zeroshot_cot_prompt =[
    {
        "role": "user",
        "content": "The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have? Let's think step-by-step"
    }
]

outputs = pipe(zeroshot_cot_prompt)
print(outputs[0]["generated_text"])


#### ToT(Tree of Thought)
# 여러가지 경로를 탐새하여 최선의 결론 선택
zeroshot_tot_prompt = [
    {
        "role": "user",
        "content": "Imagine three different experts are answering this question. All experts will write down 1 step of their thinking, then share it with the group.\
            Then all experts will go on to the next step, etc. If any expert realizes they're wrong at any point then they leave. \
                The question is 'The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have? Make sure to discuss the results."
    }
]

outputs = pipe(zeroshot_tot_prompt)
print(outputs[0]["generated_text"])


###### 출력구조에 대한 예시 제공
oneshot_template = """Create a short character profile for an RPG game. Make sure to only use this format: 
{
    "description": "A SHORT DESCRIPTION",
    "name": "THE CHARACTER'S NAME",
    "armor": "ONE PEICE OF ARMOR",
    "weapon": "ONE OR MORE WEAPONS"
}
"""

one_shot_prompt = [
    {
        "role": "user",
        "content": oneshot_template
    }
]

outputs = pipe(one_shot_prompt)
print(outputs[0]["generated_text"])


##### 제약 샘플링(특정 출력 막기)
import gc
import torch
del model, tokenizer, pipe

gc.collect()
torch.cuda.empty_cache()

from llama_cpp.llama import Llama