import os
import uuid

import chainlit as cl
from zep_cloud import FactRatingExamples, FactRatingInstruction
from chat_history_shoe_purchase import history as previous_chat_history
from dotenv import find_dotenv, load_dotenv
from openai import AsyncOpenAI

from zep_cloud.client import AsyncZep
from zep_cloud.types import Message

load_dotenv(dotenv_path=find_dotenv())

API_KEY = os.environ.get("ZEP_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

OPENAI_MODEL = "gpt-4o"

ASSISTANT_ROLE = "assistant"
USER_ROLE = "user"
USER_NAME = "Daniel"
BOT_NAME = "Assistant"

MIN_FACT_RATING = 0.5

zep = AsyncZep(api_key=API_KEY)

openai = AsyncOpenAI(api_key=OPENAI_API_KEY)

welcome_message = "Hello! How can I help you today?"

base_system_prompt = """You are a friendly assistant"""


async def load_previous_chat_history(session_id: str):
    # batch chat turns into pairs
    for i in range(0, len(previous_chat_history), 2):
        await zep.memory.add(
            session_id=session_id,
            messages=[
                Message(
                    role_type=msg["role_type"],
                    role=USER_NAME if msg["role_type"] == USER_ROLE else BOT_NAME,
                    content=msg["content"],
                )
                for msg in previous_chat_history[i : i + 2]
            ],
        )


@cl.step(name="Zep Chat History Retrieval", type="retrieval", language="python")
async def get_history(session_id: str):
    memory = await zep.memory.get(session_id=session_id, min_rating=MIN_FACT_RATING)
    message_history = []

    if memory.relevant_facts:
        message_history.append(
            {
                "role_type": "user",
                "content": "Facts about the user:\n"
                + "\n".join([f.fact for f in memory.relevant_facts]),
            }
        )

    for message in memory.messages:
        message_history.append(
            {
                "role_type": message.role_type,
                "content": message.content,
            }
        )
    return message_history


async def display_actions():
    await cl.Message(
        content="Select an action",
        actions=[
            cl.Action(
                name="Print Facts",
                value="print_facts",
            )
        ],
    ).send()


@cl.action_callback("Print Facts")
async def print_facts():
    session_id = cl.user_session.get("session_id")
    memory = await zep.memory.get(session_id=session_id, min_rating=MIN_FACT_RATING)

    if memory.relevant_facts:
        msg = cl.Message(
            author="System",
            content="\n".join([f.fact for f in memory.relevant_facts]),
        )
        await msg.send()


@cl.step(name="OpenAI", type="llm")
async def call_openai(messages):
    response = await openai.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.1,
        messages=messages,
    )
    return response.choices[0].message


@cl.on_message
async def on_message(message: cl.Message):
    session_id = cl.user_session.get("session_id")
    # Add the user's message to Zep's memory
    await zep.memory.add(
        session_id=session_id,
        messages=[
            Message(
                role_type=USER_ROLE,
                content=message.content,
                role=cl.user_session.get("user_name"),
            ),
        ],
    )

    chat_history = await get_history(session_id)

    # Load the base prompt into
    prompt = [{"role": "system", "content": base_system_prompt}]

    prompt = prompt + [
        {"role": message["role_type"], "content": message["content"]}
        for message in chat_history
    ]

    response_message = await call_openai(prompt)
    msg = cl.Message(author=BOT_NAME, content=(response_message.content))
    await msg.send()

    await zep.memory.add(
        session_id=session_id,
        messages=[
            Message(
                role_type=ASSISTANT_ROLE,
                content=response_message.content,
                role=BOT_NAME,
            ),
        ],
    )

    await display_actions()


@cl.on_chat_start
async def main():
    user_id = str(uuid.uuid4())
    session_id = str(uuid.uuid4())
    cl.user_session.set("user_id", user_id)
    cl.user_session.set("session_id", session_id)

    msg = cl.Message(author=BOT_NAME, content=welcome_message)
    await msg.send()

    await zep.user.add(
        user_id=user_id,
    )

    await zep.memory.add_session(
        user_id=user_id,
        session_id=session_id,
        fact_rating_instruction=FactRatingInstruction(
            instruction="Rate the facts by how relevant they are to purchasing shoes.",
            examples=FactRatingExamples(
                high="The user has agreed to purchase a Reebok running shoe.",
                medium="The user prefers running to cycling.",
                low="The user purchased a dress.",
            ),
        ),
    )

    await load_previous_chat_history(session_id)
