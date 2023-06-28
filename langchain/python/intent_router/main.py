from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ZepChatMessageHistory, ConversationBufferMemory

from intent_router import IntentRouterChain, IntentModel

ZEP_API_URL = "http://localhost:8000"

sales_template = """You are widgets sales rep and your job is to assist humans with completing the purchase of a widget.
In order to close a widget sale, you need to know how many widgets the human would like to purchase. Don't be pushy about making the sale,
but remember that your job is dependent on achieving your sales quota.

Important sales information:
- The current price of widgets is $499.
- If a customer wants to purchase 10 or more widgets, you can offer a 20% discount. There are no discounts available for smaller purchases.
- They come in blue, black, hot pink, and sparkling gold colors.
- Widgets have a warranty of 1 year.

Here are the prior messages in this conversation:
{chat_history}

Here is a question: {input}
"""

support_template = """You are support agent for a widget producer and your job is to assist humans with issues they may have with using the widgets
they purchased from us.
In order to assist a human, you need to know when the widget was purchased, what color it is, and what the user's support issue is.

Important notes about working with humans:
- Always be friendly! They may be upset if their widget doesn't work. Being friendly will help you getting the answers you need.
- Humans can only answer 1 question at a time.

Important support information:
- Widgets have a warranty of 1 year.
- Sparkling gold widgets tend to flake the sparkling paint. We are happy to replace these as long as the widget is under warranty.
- Many humans forget to power on their widgets before attempting use. This should be your first line of questioning.

Today's date is 06/27/2023.

Here are the prior messages in this conversation:
{chat_history}

Here is a question: {input}
"""


def main():
    load_dotenv()

    zep_chat_history = ZepChatMessageHistory(session_id="test", url=ZEP_API_URL)
    memory = ConversationBufferMemory(chat_memory=zep_chat_history, memory_key="chat_history")

    intent_models = [
        IntentModel(
            intent="purchase a widget",
            description="the human would like to make a purchase",
            prompt=sales_template,
            default=True,
        ),
        IntentModel(
            intent="needs customer support",
            description="the human has a support query",
            prompt=support_template,
        ),
    ]

    llm = ChatOpenAI(model_name="gpt-3.5-turbo")

    chain = IntentRouterChain.from_intent_models(
        intent_models=intent_models,
        llm=llm,
        embedding_model=OpenAIEmbeddings(),
        memory=memory,
        verbose=True,
    )

    print(chain.run({"input": "I'm upset my widget doesn't work!"}))


if __name__ == "__main__":
    main()
