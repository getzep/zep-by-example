from __future__ import annotations

import asyncio
import time
from typing import Any, Dict
from typing import Type
from uuid import uuid4

from dotenv import load_dotenv
from langchain import LLMChain
from langchain.callbacks.manager import (
    atrace_as_chain_group,
)
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.memory import ZepMemory
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import BaseMemory

from chat_history import history

TEMPERATURE = 0.0
ZEP_API_URL = "http://localhost:8000"
PROMPT_TEMPLATE = """You are very knowledgeable about the world and enjoy sharing your 
knowledge with others. Please answer the human's question. Limit your answers to a 
maximum of three sentences.

Here are the prior messages in this conversation:
{chat_history}

Here is a question: {input}
"""


async def memory_test(
    run_tag_prefix: str,
    memory_class: Type[BaseMemory],
    memory_args: Dict[str, Any],
    runs: int = 5,
    prompt_template: str = PROMPT_TEMPLATE,
):
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["input", "chat_history"]
    )
    for i in range(runs):
        run_tag = f"{run_tag_prefix}_{i}"

        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo", tags=[run_tag], temperature=TEMPERATURE
        )

        if memory_class.__name__ == "ZepMemory":
            memory = memory_class(memory_key="chat_history", **memory_args)
        else:
            memory = memory_class(llm=llm, memory_key="chat_history", **memory_args)

        chain = LLMChain(llm=llm, memory=memory, prompt=prompt, verbose=True)

        async with atrace_as_chain_group(run_tag) as async_group_manager:
            for msg in history:
                if msg["role"] == "ai":
                    continue
                print(
                    await chain.arun(
                        {"input": msg["content"]}, callbacks=async_group_manager
                    )
                )
                time.sleep(1)


async def summary_memory_test(mem_token_limit: int, runs: int = 5):
    run_tag_prefix = f"sm_{mem_token_limit}"
    mem_args = {"max_token_limit": mem_token_limit}

    await memory_test(
        run_tag_prefix=run_tag_prefix,
        memory_class=ConversationSummaryBufferMemory,
        memory_args=mem_args,
        runs=runs,
    )


async def zep_memory_test(runs: int = 5):
    run_tag_prefix = "zep"
    session_id = str(uuid4())
    mem_args = {
        "session_id": session_id,
        "url": ZEP_API_URL,
    }

    await memory_test(
        run_tag_prefix=run_tag_prefix,
        memory_class=ZepMemory,
        memory_args=mem_args,
        runs=runs,
    )


async def test_suite():
    runs = 1
    await summary_memory_test(mem_token_limit=250, runs=runs)
    await summary_memory_test(mem_token_limit=500, runs=runs)
    await zep_memory_test(runs=runs)


def main():
    load_dotenv()

    asyncio.run(test_suite())


if __name__ == "__main__":
    main()
