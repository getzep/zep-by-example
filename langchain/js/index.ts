import 'dotenv/config'
import {ChatOpenAI} from "langchain/chat_models/openai";
import {ConversationChain} from "langchain/chains";
import {ZepMemory} from "langchain/memory/zep";
import {ZepRetriever} from "langchain/retrievers/zep";

import {v4 as uuidv4} from "uuid";
import {Memory, Message} from "@getzep/zep-js";

(async () => {
    // Example session ID
    const sessionId = uuidv4();
    const zepURL = "http://localhost:8000";

    const memory = new ZepMemory({
        sessionId,
        baseURL: zepURL,
    });

    // extract the ZepClient from the Memory instance. We'll use this later
    // to persist a chat history and inspect the Zep memory for this session.
    const {zepClient} = memory;

    const model = new ChatOpenAI({
        modelName: "gpt-3.5-turbo",
        temperature: 0,
    });

    const history = [
        {
            "role": "human",
            "content": "Who was Octavia Butler?"
        },
        {
            "role": "ai",
            "content": (
                "Octavia Estelle Butler (June 22, 1947 – February 24, 2006) was an American" +
                " science fiction author."
            ),
        },
        {"role": "human", "content": "Which books of hers were made into movies?"},
        {
            "role": "ai",
            "content": (
                "The most well-known adaptation of Octavia Butler's work is the FX series" +
                " Kindred, based on her novel of the same name."
            ),
        },
        {"role": "human", "content": "Who were her contemporaries?"},
        {
            "role": "ai",
            "content": (
                "Octavia Butler's contemporaries included Ursula K. Le Guin, Samuel R." +
                " Delany, and Joanna Russ."
            ),
        },
        {"role": "human", "content": "What awards did she win?"},
        {
            "role": "ai",
            "content": (
                "Octavia Butler won the Hugo Award, the Nebula Award, and the MacArthur" +
                " Fellowship."
            ),
        },
        {
            "role": "human",
            "content": "Which other women sci-fi writers might I want to read?",
        },
        {
            "role": "ai",
            "content": "You might want to read Ursula K. Le Guin or Joanna Russ.",
        },
        {
            "role": "human",
            "content": (
                "Write a short synopsis of Butler's book, Parable of the Sower. What is it" +
                " about?"
            ),
        },
        {
            "role": "ai",
            "content": (
                "Parable of the Sower is a science fiction novel by Octavia Butler," +
                " published in 1993. It follows the story of Lauren Olamina, a young woman" +
                " living in a dystopian future where society has collapsed due to" +
                " environmental disasters, poverty, and violence."
            ),
        },
    ];
    const messages = history.map(
        ({role, content}) => new Message({role, content})
    );
    const zepMemory = new Memory({messages});

    await zepClient.addMemory(sessionId, zepMemory);


    const chain = new ConversationChain({llm: model, memory});

    const res1 = await chain.call({input: "Is there a sequel Parable of the Sower?"});
    console.log({res1, memory: await memory.loadMemoryVariables({})});
    /*
{
  res1: {
    response: 'Yes, there is a sequel to Parable of the Sower called Parable of the Talents, which was published in 1998. It continues the story of Lauren Olamina and her community as they face new challenges and struggles in their post-apocalyptic world.'
  },
  memory: {
    history: 'human: Who were her contemporaries?\n' +
      "ai: Octavia Butler's contemporaries included Ursula K. Le Guin, Samuel R. Delany, and Joanna Russ.\n" +
      'human: What awards did she win?\n' +
      'ai: Octavia Butler won the Hugo Award, the Nebula Award, and the MacArthur Fellowship.\n' +
      'human: Which other women sci-fi writers might I want to read?\n' +
      'ai: You might want to read Ursula K. Le Guin or Joanna Russ.\n' +
      "human: Write a short synopsis of Butler's book, Parable of the Sower. What is it about?\n" +
      'ai: Parable of the Sower is a science fiction novel by Octavia Butler, published in 1993. It follows the story of Lauren Olamina, a young woman living in a dystopian future where society has collapsed due to environmental disasters, poverty, and violence.\n' +
      'Human: Is there a sequel Parable of the Sower?\n' +
      'AI: Yes, there is a sequel to Parable of the Sower called Parable of the Talents, which was published in 1998. It continues the story of Lauren Olamina and her community as they face new challenges and struggles in their post-apocalyptic world.'
  }
}
    */

    const res2 = await chain.call({input: "Please recommend books similar to Parable of the Sower."});
    console.log({res2, memory: await memory.loadMemoryVariables({})});

    /*
{
  res2: {
    response: "You might enjoy reading The Hunger Games by Suzanne Collins, The Road by Cormac McCarthy, or The Handmaid's Tale by Margaret Atwood."
  },
  memory: {
    history: 'human: What awards did she win?\n' +
      'ai: Octavia Butler won the Hugo Award, the Nebula Award, and the MacArthur Fellowship.\n' +
      'human: Which other women sci-fi writers might I want to read?\n' +
      'ai: You might want to read Ursula K. Le Guin or Joanna Russ.\n' +
      "human: Write a short synopsis of Butler's book, Parable of the Sower. What is it about?\n" +
      'ai: Parable of the Sower is a science fiction novel by Octavia Butler, published in 1993. It follows the story of Lauren Olamina, a young woman living in a dystopian future where society has collapsed due to environmental disasters, poverty, and violence.\n' +
      'Human: Is there a sequel Parable of the Sower?\n' +
      'AI: Yes, there is a sequel to Parable of the Sower called Parable of the Talents, which was published in 1998. It continues the story of Lauren Olamina and her community as they face new challenges and struggles in their post-apocalyptic world.\n' +
      'Human: Please recommend books similar to Parable of the Sower.\n' +
      "AI: You might enjoy reading The Hunger Games by Suzanne Collins, The Road by Cormac McCarthy, or The Handmaid's Tale by Margaret Atwood."
  }
}
    */

    // Zep's Extractors run asynchronously. We'll naively wait for them to complete by sleeping for a second.
    console.log("Sleeping for 1 seconds...");
    sleep(1000); // Sleep for 3 seconds
    console.log("Done sleeping!");

    // Use the ZepRetriever to search for relevant messages
    const retriever = new ZepRetriever({sessionId, url: zepURL})
    const relevantResults = await retriever.getRelevantDocuments("I enjoy dystopian sci-fi novels.")
    relevantResults.forEach(element => {
        if (element.metadata.dist >= 0.8) {
            console.log(`${element.pageContent} => ${element.metadata.dist}`);
        }
    });
    /*
    You might enjoy reading The Hunger Games by Suzanne Collins, The Road by Cormac McCarthy, or The Handmaid's Tale by Margaret Atwood. => 0.8598901744352312
Which other women sci-fi writers might I want to read? => 0.8353206100397812
You might want to read Ursula K. Le Guin or Joanna Russ. => 0.8236142000294512
Octavia Butler's contemporaries included Ursula K. Le Guin, Samuel R. Delany, and Joanna Russ. => 0.8089540052202283
Parable of the Sower is a science fiction novel by Octavia Butler, published in 1993. It follows the story of Lauren Olamina, a young woman living in a dystopian future where society has collapsed due to environmental disasters, poverty, and violence. => 0.8052832541478573
Octavia Butler won the Hugo Award, the Nebula Award, and the MacArthur Fellowship. => 0.8038596194093931
     */

    // View the most recent summary for this SessionID
    const serverMemory = await zepClient.getMemory(sessionId);
    console.log(serverMemory?.summary)
    /*
 Summary {
  uuid: '0ab0002e-84c1-4d68-ab8b-952bf169b3be',
  created_at: '2023-05-31T01:11:24.792728Z',
  content: "The human asks about Octavia Butler and the AI identifies her as an American science fiction author. The human then inquires about which of her books were made into movies and the AI mentions the FX series Kindred. The human also asks about Butler's contemporaries and the AI names Ursula K. Le Guin, Samuel R. Delany, and Joanna Russ.",
  recent_message_uuid: '90bf11fc-359b-456b-afa1-d498964f8212',
  token_count: 393
}
     */

    // View the metadata, including Named Entities for the last message.
    const lastMessage = serverMemory?.messages[serverMemory?.messages.length - 1];
    console.log(JSON.stringify(lastMessage));
    /*
{
  "uuid": "5af9edbb-6aa8-40f5-917b-4a088bb815e4",
  "created_at": "2023-05-31T01:11:26.71254Z",
  "role": "AI",
  "content": "You might enjoy reading The Hunger Games by Suzanne Collins, The Road by Cormac McCarthy, or The Handmaid's Tale by Margaret Atwood.",
  "token_count": 31,
  "metadata": {
    "system": {
      "entities": [
        {
          "Label": "WORK_OF_ART",
          "Matches": [
            {
              "End": 40,
              "Start": 24,
              "Text": "The Hunger Games"
            }
          ],
          "Name": "The Hunger Games"
        },
        {
          "Label": "PERSON",
          "Matches": [
            {
              "End": 59,
              "Start": 44,
              "Text": "Suzanne Collins"
            }
          ],
          "Name": "Suzanne Collins"
        },
        {
          "Label": "WORK_OF_ART",
          "Matches": [
            {
              "End": 88,
              "Start": 61,
              "Text": "The Road by Cormac McCarthy"
            }
          ],
          "Name": "The Road by Cormac McCarthy"
        },
        {
          "Label": "WORK_OF_ART",
          "Matches": [
            {
              "End": 112,
              "Start": 93,
              "Text": "The Handmaid's Tale"
            }
          ],
          "Name": "The Handmaid's Tale"
        },
        {
          "Label": "PERSON",
          "Matches": [
            {
              "End": 131,
              "Start": 116,
              "Text": "Margaret Atwood"
            }
          ],
          "Name": "Margaret Atwood"
        }
      ]
    }
  }
}
    */

})();

function sleep(ms: number) {
    const date = Date.now();
    let currentDate = 0;
    do {
        currentDate = Date.now();
    } while (currentDate - date < ms);
}
