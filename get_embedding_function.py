from langchain_ollama import OllamaEmbeddings

def get_embedding_function():
    return OllamaEmbeddings("deepseek-r1:1.5b")

embedding_function = get_embedding_function()
test_embedding = embedding_function.embed_query("test")
print(len(test_embedding))