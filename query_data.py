
import argparse
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer questions strictly based on the provided context.

Context:
{context}

Question: {question}

Format your response like this:
**Answer:** <your answer here>
"""


def query_rag(query_text: str, category: str = None):  # Fixed parameter name
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, 
                embedding_function=embedding_function)
    
    search_filter = {"category": category} if category else None
    
    results = db.similarity_search_with_score(
        query_text,
        k=3 if category else 5,
        filter=search_filter,
        #score_threshold=0.3
    )
    
    context_items = [doc.page_content for doc, _ in results]
    context_text = "\n\n".join(context_items)
    
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
        context=context_text, 
        question=query_text
    )
    
    model = OllamaLLM(model="deepseek-r1:1.5b")
    response = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response}\nSources: {sources}"
    print(formatted_response)
    return formatted_response

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    parser.add_argument("--category", type=str, help="Filter by category")
    args = parser.parse_args()
    query_rag(args.query_text, args.category)

if __name__ == "__main__":
    main()

