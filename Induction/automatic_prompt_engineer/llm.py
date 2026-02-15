"""Contains classes for querying large language models."""
import os
import time
from tqdm import tqdm
from abc import ABC, abstractmethod
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM
import openai
import torch
import asyncio
from typing import Any
from automatic_prompt_engineer import template
from openai import OpenAI, AsyncOpenAI
import random

gpt_costs_per_thousand = {
    'davinci': 0.0200,
    'curie': 0.0020,
    'babbage': 0.0005,
    'ada': 0.0004
}

async def dispatch_openai_requests(
    messages_list: list[list[dict[str,Any]]],
    model: str,
    temperature: float,
    max_tokens: int,
    frequency_penalty: int,
    presence_penalty: int
) -> list[str]:
    """Dispatches requests to OpenAI API asynchronously.
    
    Args:
        messages_list: List of messages to be sent to OpenAI ChatCompletion API.
        model: OpenAI model to use.
        temperature: Temperature to use for the model.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use for the model.
    Returns:
        List of responses from OpenAI API.
    """
    client = AsyncOpenAI()
    async_responses = []
    for messages in messages_list:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            seed=0
        )
        async_responses.append(response)

    completed_responses = await asyncio.gather(*async_responses)
    return [response.choices[0].message.content for response in completed_responses]


def model_from_config(config, disable_tqdm=True):
    """Returns a model based on the config."""
    model_type = config["name"]
    if model_type == "GPT_forward":
        return GPT_Forward(config, disable_tqdm=disable_tqdm)
    elif model_type == "GPT_insert":
        return GPT_Insert(config, disable_tqdm=disable_tqdm)
    elif model_type == "Llama_Forward":
        return Llama_Forward(config, disable_tqdm=disable_tqdm)
    raise ValueError(f"Unknown model type: {model_type}")


class LLM(ABC):
    """Abstract base class for large language models."""

    @abstractmethod
    def generate_text(self, prompt):
        """Generates text from the model.
        Parameters:
            prompt: The prompt to use. This can be a string or a list of strings.
        Returns:
            A list of strings.
        """
        pass

MODEL_DIR = {'mistral': "mistralai/Mistral-7B-Instruct-v0.2",
             'llama': "meta-llama/Meta-Llama-3.1-8B-Instruct",
             'qwen': "Qwen/Qwen2.5-7B-Instruct",
             'internlm': "internlm/internlm3-8b-instruct",
             'falcon': "tiiuae/Falcon3-7B-Instruct",
             'gemma': "google/gemma-7b-it",
             'yi': "01-ai/Yi-1.5-6B-Chat",
             'phi': "microsoft/Phi-3-small-8k-instruct",
             'vicuna': "lmsys/vicuna-13b-v1.3"
             }

def get_input_template(api_model):
    system_prompts ={
        'mistral': "You are an AI assistant that provides concise and accurate answers.",
        'llama' : "You are a helpful assistant.",
        'qwen' : "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        'internlm' : """You are an AI assistant whose name is InternLM (书生·浦语).
        - InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
        - InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.""",
        'falcon' : "You are a helpful friendly assistant Falcon3 from TII, try to follow instructions as much as possible.",
        'gpt': "You are a helpful assistant."        
    }
    if api_model == 'internlm':
        messages = [
            {"role": "assistant", "content": system_prompts[api_model]},
            {"role": "user", "content": "[PROMPT]"}
        ]
    elif api_model in ['gemma', 'yi', 'phi']:
        messages = [
            {"role": "user", "content": "[PROMPT]"}
        ]
    else:
        messages = [
            {"role": "system", "content": system_prompts[api_model]},
            {"role": "user", "content": "[PROMPT]"}
        ]
    message_template = template.MessageTemplate(messages)
    return message_template

def get_or_create_event_loop():
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop

class Black_box(LLM):
    def __init__(self, config, api_model, needs_confirmation=False, disable_tqdm=True):
        """Initializes the model."""
        self.api_model = api_model
        self.config = config
        self.needs_confirmation = needs_confirmation
        self.disable_tqdm = disable_tqdm
 
        self.template = get_input_template(api_model)
        self.loop = asyncio.get_event_loop() if asyncio.get_event_loop().is_running() else asyncio.new_event_loop()

    def generate_text(self, prompts):
        return self.loop.run_until_complete(self._generate_text_async(prompts))

    async def _generate_text_async(self, prompts):
        if not isinstance(prompts, list):
            prompts = [prompts]
        prompts = [self.template.fill(prompt) for prompt in prompts]
        batch_size = min(len(prompts), 20)
        prompt_batches = [prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]

        text = []
        for prompt_batch in prompt_batches:
            text += await self.__async_generate(prompt_batch)
        return text

    async def __async_generate(self, prompt_batch):
            answer = None
            while answer is None:
                try:
                    predictions = await asyncio.wait_for(
                        dispatch_openai_requests(
                            messages_list=prompt_batch,
                            model="gpt-4.1",
                            temperature=0,
                            max_tokens=64,
                            frequency_penalty=0,
                            presence_penalty=0
                        ),
                        timeout=20
                    )
                    answer = predictions
                except asyncio.TimeoutError:
                    print("The task exceeded the time limit 20 s.")
                except Exception as e:
                    print(e)
                    print("Retrying....")
                    await asyncio.sleep(20)
            return answer