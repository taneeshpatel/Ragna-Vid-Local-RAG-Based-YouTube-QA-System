from rag.retriever import retrieve_relevant_chunks
from rag.qa_engine import generate_answer

question = input("❓ Ask a question about the video: ")

# Step 1: Retrieve relevant transcript chunks
top_chunks = retrieve_relevant_chunks(question, top_k=3)
context = "\n\n".join(top_chunks)

# Step 2: Generate answer from LLM
answer = generate_answer(context, question)

print("\n🤖 Answer:\n", answer)
