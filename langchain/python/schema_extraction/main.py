from __future__ import annotations

from dotenv import load_dotenv
from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI
from pydantic import Field

from schema_extractor_memory import SchemaExtractorMemory, ExtractorSchema


class Address(ExtractorSchema):
    street: str | None = Field(description="the street address", max_length=100)
    city: str | None = Field(description="the city", max_length=100)
    state: str | None = Field(description="the state", max_length=100)
    zip: str | None = Field(
        description="the zip code",
        max_length=20,
    )


class Person(ExtractorSchema):
    first_name: str | None = Field(description="the human's first name", max_length=100)
    last_name: str | None = Field(description="the human's last name", max_length=100)
    email: str | None = Field(
        description="the human's email address",
        max_length=100,
    )
    phone: str | None = Field(description="the human's phone number", max_length=20)


class OrderItem(ExtractorSchema):
    size: str | None = Field(
        description="the size of the shoe",
    )
    color: str | None = Field(
        description="the color of the shoe",
    )
    brand: str | None = Field(
        description="the brand of the shoe",
    )
    quantity: str | None = Field(
        default="1",
        description="the number of shoes ordered. assume 1 unless otherwise specified",
    )
    style: str | None = Field(
        description="the style of the shoe",
    )


class Order(ExtractorSchema):
    person: Person | None = Field(description="the person ordering the shoes")
    item: OrderItem | None = Field(description="the item being ordered")
    shipping_address: Address | None = Field(
        description="the person's shipping address"
    )


def main():
    load_dotenv()

    llm_chat = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
    llm_extraction = ChatOpenAI(temperature=0, model="gpt-4-0613")

    memory = SchemaExtractorMemory(
        model_schema=Order(),
        llm=llm_extraction,
        input_key="input",
        memory_key="chat_history",
    )

    sales_template = """
You are ashoe sales rep and your job is to assist humans with completing the
purchase of a shoe.

In order to close a shoe sale, you need to know:
- the shoe style and size of shoe
- full name
- the human's email address
- the human's phone number
- the human's shipping address

IMPORTANT INSTRUCTIONS:
- You may only ask for one piece of information at a time. 
- Ensure that you collect all of the above information in order to close the sale. 
- Confirm the order details with the human before closing the sale.

This is what you already know about this order:
{order_details}

Here are the prior messages in this conversation:
{chat_history}

Human's response: {input}
"""

    llm_chain = LLMChain(
        llm=llm_chat,
        prompt=PromptTemplate.from_template(sales_template),
        memory=memory,
        verbose=True,
    )

    human_messages = [
        "hello, I'm Jane!",
        "I'd like to buy a pair of Puma Suede Classics.",
        "I'd prefer them to be black.",
        "Yes. I'm a size 9",
        "Jane Austin",
        "jane@sanditon.com",
        "415-555-1234",
        "555 Main St, San Francisco, CA 94555",
        "Yes, that's correct.",
        "No thank you.",
    ]

    for message in human_messages:
        print(
            llm_chain.run(
                {
                    "input": message,
                    "order_details": memory.model.dict(exclude_unset=True),
                }
            )
        )

    print("MODEL: >>>>", memory.model)


if __name__ == "__main__":
    main()


# TODOs:
# - deal with conflicts when same data is extracted from multiple messages
# - generate instructions from model schema
# - explicitly identify missing data
# - explicitly prompt a confirm
# - validate the data!
