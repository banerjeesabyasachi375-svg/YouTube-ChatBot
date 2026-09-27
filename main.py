from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables (.env)
load_dotenv()

# ==========================================
# STEP 1: Fetch YouTube Transcript
# ==========================================
video_id = input("Provide Youtube Video Id:")

yt_api = YouTubeTranscriptApi()
transcript_data = yt_api.fetch(video_id)

# Access dictionary items safely
whole_transcript = " ".join([item.text for item in transcript_data])


# ==========================================
# STEP 2: Chunk Transcript into Documents
# ==========================================
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

docs = splitter.create_documents([whole_transcript])
print(f"Generated {len(docs)} document chunks.")


# ==========================================
# STEP 3: Embeddings & Vector DB
# ==========================================
embedding = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embedding,
    persist_directory="data/chroma",
    collection_name="demo"
)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2}
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ==========================================
# STEP 4: History-Aware Retriever & Chain Setup
# ==========================================

# 1. Prompt to reformulate follow-up questions using conversation history
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given a chat history and the latest user question, "
               "formulate a standalone question that can be understood without the chat history. "
               "Do NOT answer the question, just reformulate it if needed, otherwise return it as is."),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}")
])

history_aware_query_chain = contextualize_q_prompt | llm | RunnableLambda(lambda res: res.content)

# Helper function to format retrieved documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Retrieve documents conditionally based on history
def get_context(inputs):
    if inputs["chat_history"]:
        standalone_q = history_aware_query_chain.invoke(inputs)
    else:
        standalone_q = inputs["question"]
    docs = retriever.invoke(standalone_q)
    return format_docs(docs)

# 2. Main Q&A Prompt with history and context
final_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer questions using ONLY the provided transcript context.\n\nContext:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}")
])

# Complete LCEL Chain
chain = (
    RunnableParallel({
        "context": RunnableLambda(get_context),
        "chat_history": lambda x: x["chat_history"],
        "question": lambda x: x["question"]
    })
    | final_prompt
    | llm
)


# ==========================================
# STEP 5: Interactive Infinite Loop
# ==========================================
if __name__ == "__main__":
    chat_history = []
    print("\n--- RAG Transcript Chat initialized. Type 'exit' or 'quit' to stop ---\n")

    while True:
        user_question = input("Question: ").strip()

        if not user_question:
            continue
        if user_question.lower() in ["exit", "quit"]:
            print("Ending conversation session. Goodbye!")
            break

        # Invoke chain with the user input and persistent chat history
        response = chain.invoke({
            "question": user_question,
            "chat_history": chat_history
        })

        print(f"\nAssistant: {response.content}\n")

        # Append messages to maintain history across turns
        chat_history.append(HumanMessage(content=user_question))
        chat_history.append(AIMessage(content=response.content))
