from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chat_models import init_chat_model
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaEmbeddings, ChatOllama

from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

video_id = "aircAruvnKk" # only the ID, not full URL
try:
    # Fetch the English transcript
    transcript = YouTubeTranscriptApi().fetch(
        video_id,
        languages=["en"]
    )

    # Convert transcript snippets into plain text
    transcript_text = " ".join(
        snippet.text for snippet in transcript.snippets
    )

    # print(transcript_text)

except TranscriptsDisabled:
    print("No captions available for this video.")
except Exception as e:
    print(f"Error fetching transcript: {e}")

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript_text])

# for i in chunks:
#     print(i,end="==========================================================================================")

embeddings = OllamaEmbeddings(
    model="nomic-embed-text:latest"
)

vector_store = FAISS.from_documents(chunks, embeddings)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})



#augmentation
llm = ChatOllama(
    model="dagbs/qwen2.5-coder-7b-instruct-abliterated:q4_0 ",
    temperature=0
)
prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)

question          = "what is the capital of newyork"
retrieved_docs    = retriever.invoke(question)


#context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
#final_prompt = prompt.invoke({"context": context_text, "question": question})

#answer = llm.invoke(final_prompt)


def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

parser = StrOutputParser()

main_chain = parallel_chain | prompt | llm | parser

outpt = main_chain.invoke('what do you think what are the mian points discussed tell me in 3 bullet points with evidance')

print(outpt)