/**
 * This example shows how to use Zep with LangChain.js to create a simple RAG application.
 *
 * In this example, we'll be using both ZepMemory and ZepVectorStore, providing our app
 * with conversational memory and the ability to search through a document collection.
 *
 * We'll be using LangSmith for observability. If you'd like to do the same, you'll need
 * a LangSmith account and to set the following keys in your environment.
 *
 * We'll be using dotenv to load our environment variables from a .env file.
 *
 * export LANGCHAIN_TRACING_V2=true
 * export LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
 * export LANGCHAIN_API_KEY=<your-api-key>
 * export LANGCHAIN_PROJECT=<your-project>  # if not specified, defaults to "default"
 */
import "dotenv/config";
import { IZepConfig, ZepVectorStore } from "langchain/vectorstores/zep";
import { Document } from "langchain/document";
import { FakeEmbeddings } from "langchain/embeddings/fake";
import { randomUUID } from "crypto";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { ZepMemory } from "langchain/memory/zep";
import { ChatOpenAI } from "langchain/chat_models/openai";
import { history } from "./history.js";
import { Memory, Message, ZepClient } from "@getzep/zep-js";
import {
  ConversationalRetrievalQAChain,
  ConversationChain,
} from "langchain/chains";
import { ZepRetriever } from "langchain/retrievers/zep";
import { DynamicTool } from "langchain/tools";
import { initializeAgentExecutorWithOptions } from "langchain/agents";

const ZEP_API_URL = process.env.ZEP_API_URL || "http://localhost:8000";
const ZEP_API_KEY = process.env.ZEP_API_KEY;

// The name of the Zep collection we'll be using. We'll be creating a new collection for this example.
const ZEP_COLLECTION_NAME = `scifi${randomUUID().split("-").join("")}`;

const ZEP_COLLECTION_CONFIG: IZepConfig = {
  apiUrl: ZEP_API_URL,
  apiKey: ZEP_API_KEY,
  collectionName: ZEP_COLLECTION_NAME,
  embeddingDimensions: 1536, // Set to the width of the model configured in Zep. Use 1536 for OpenAI
  isAutoEmbedded: true,
};

async function loadDocs(path: string): Promise<Document[]> {
  return new DirectoryLoader(path, {
    ".txt": (path) => new TextLoader(path),
  }).load();
}

async function loadDocsIntoVectorStore(
  config: IZepConfig,
): Promise<ZepVectorStore> {
  const docs = await loadDocs("./books");
  console.log(`Loaded ${docs.length} documents`);

  // Split the documents into chunks
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
    separators: ["\n\n", "\n", " ", "", "\r", "\r\n"], // add carriage returns to the list of separators
  });

  // Split the documents into chunks. We also add the source of the document as a header to each chunk.
  const chunks = (
    await Promise.all(
      docs.map((doc) =>
        splitter.splitDocuments([doc], {
          chunkHeader: doc.metadata.source
            ? "SOURCE: " + doc.metadata.source.split("/").pop() + "\n\n"
            : "",
          appendChunkOverlapHeader: true,
        }),
      ),
    )
  ).flat();

  // We're creating a new Document Collection and so need to pass in the Zep collection config
  // Note that we're using a fake embedding model here. Zep will be doing the embedding for us, which allows us to use
  // local, low-latency embedding models. You can also set isAutoEmbedded to false and embed the d
  // ocuments using LangChain's embedding models before passing them to Zep.
  return ZepVectorStore.fromDocuments(chunks, new FakeEmbeddings(), config);
}

async function createZepMemoryRetriever(
  sessionId: string,
): Promise<ZepRetriever> {
  return new ZepRetriever({
    url: ZEP_API_URL,
    apiKey: ZEP_API_KEY,
    sessionId: sessionId,
  });
}

async function getPeopleFromMemoryTool(
  query: string,
  retriever: ZepRetriever,
): Promise<string> {
  return retriever
    .getRelevantDocuments(query, {
      metadata: {
        where: { jsonpath: '$.system.entities[*] ? (@.Label == "PERSON")' },
      },
    })
    .then((docs) => {
      const filteredDocs = docs.filter((doc) => doc.metadata.dist >= 0.8);
      return (
        filteredDocs.length > 0
          ? filteredDocs.map((doc) => doc.pageContent)
          : ["No results"]
      ).join("\n\n");
    });
}

async function getBookSearchTool(
  query: string,
  vectorStore: ZepVectorStore,
): Promise<string> {
  const retriever = await vectorStore.asRetriever({ searchType: "mmr", k: 4 });
  return retriever.getRelevantDocuments(query).then((docs) => {
    return docs.map((doc) => doc.pageContent).join("\n\n");
  });
}

(async () => {
  // Create a new ZepClient instance
  const client = await ZepClient.init(ZEP_API_URL, ZEP_API_KEY);

  // Create a session ID for our conversation. This ID could represent our user, or a
  // conversation thread with a user. i.e. You can map multiple sessions to a single user
  // in your data model.
  const sessionId = randomUUID();

  // add the sample chat history to the Zep memory
  const messages = history.map(
    ({ role, content }: { role: string; content: string }) =>
      new Message({ role, content }),
  );
  const zepMemory = new Memory({ messages });
  await client.memory.addMemory(sessionId, zepMemory);

  // sleep for 2 seconds to allow Zep to summarize the memory
  await new Promise((resolve) => setTimeout(resolve, 2000));

  // Create a new ChatOpenAI model instance. We'll use this for both oru chain and agent.
  const model = new ChatOpenAI({
    modelName: "gpt-3.5-turbo",
    temperature: 0,
  });

  // Let's create a new ZepMemory instance with very simple configuration.
  // We'll use this in our first chain to demonstrate the basics by recalling the
  // chat history we've just added to Zep.
  const memorySimple = new ZepMemory({
    sessionId,
    baseURL: ZEP_API_URL,
    apiKey: ZEP_API_KEY,
  });

  // Let's start with a simple chain and ask the LLM what we've discussed so far.
  const conversationChain = new ConversationChain({
    llm: model,
    memory: memorySimple,
  });
  const res1 = await conversationChain.run("What have we discussed so far?");
  console.log(res1);

  // Let's create a new ZepMemory instance. This will be used to store the current state of the conversation.
  // Zep will also auto-summarize and enrich memories for you.
  const memory = new ZepMemory({
    sessionId,
    baseURL: ZEP_API_URL,
    apiKey: ZEP_API_KEY,
    memoryKey: "chat_history",
    inputKey: "question", // The key for the input to the chain
    outputKey: "text", // The key for the final conversational output of the chain
  });

  // Create a new ZepVectorStore instance and load a document collection into it
  const vectorStore = await loadDocsIntoVectorStore(ZEP_COLLECTION_CONFIG);

  // Wait for the ZepVectorStore to finish embedding the documents
  console.log("Waiting for Zep to finish embedding documents...");
  while (true) {
    const c = await client.document.getCollection(ZEP_COLLECTION_NAME);
    console.log(
      `Embedding status: ${c.document_embedded_count}/${c.document_count} documents embedded`,
    );
    await new Promise((resolve) => setTimeout(resolve, 1000));
    if (c.status === "ready") {
      break;
    }
  }

  // Create a new ConversationalRetrievalQAChain instance
  // Initialize the chain with the model, the vector store, and the memory
  // We'll configure the VectorStore's Retriever to use Maximal Marginal Relevance reranking.
  // This will re-rank the results of the search to ensure that the results are diverse.
  const chain = ConversationalRetrievalQAChain.fromLLM(
    model,
    vectorStore.asRetriever({
      searchType: "mmr",
      k: 4,
    }),
    { memory: memory },
  );

  const questions = [
    "Can you tell me about a Philip K. Dick book?",
    "What is the wub?",
    "Have any authors written about visiting the moon??",
    "Who were the Fords in Vonnegut's book, The Big Trip Up Yonder?",
  ];

  // Ask the questions!
  for (const question of questions) {
    const res = await chain.call({ question: question });
    console.log("Result: " + res.text);
  }

  // Let's inspect the session's Memory
  console.log(
    "Memory: " +
      JSON.stringify(await client.memory.getMemory(sessionId), null, 2),
  );

  // Let's build an agent!
  const zepMemoryRetriever = await createZepMemoryRetriever(sessionId);

  // Create some tools.
  const tools = [
    new DynamicTool({
      name: "peopleRetriever",
      description: `call this if you want to search for authors, characters, or people we may have discussed in the 
past. input should be a search string`,
      func: async (query) =>
        await getPeopleFromMemoryTool(query, zepMemoryRetriever),
    }),
    new DynamicTool({
      name: "bookSearch",
      description:
        "call this if to search for passages in sci-fi books. input should be a search string",
      func: async (query) => await getBookSearchTool(query, vectorStore),
    }),
  ];

  const executor = await initializeAgentExecutorWithOptions(tools, model, {
    agentType: "zero-shot-react-description",
  });

  const questions2 = [
    "Have we discussed Kurt Vonnegut?",
    "What books has Philip K. Dick written?",
    "Which sci-fi novel featured a gun club headquartered in Baltimore?",
    "Have we discussed President Biden?",
  ];

  // Ask the questions!
  for (const question of questions2) {
    const res = await executor.call({ input: question });
    console.log("Result: " + res.output);
  }
})();
