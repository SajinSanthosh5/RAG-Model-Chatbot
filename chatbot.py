import ollama
import numpy as np
embedded_model="http://hf.co/CompendiumLabs/bge-base-en-v1.5-gguf"
language_model="http://hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF"
vector_DB=[]
def add_chuck_to_database(chunk):
    try:
        response=ollama.embed(model=embedded_model,input=chunk)
        if "embedding" in response:
            embedding=response["embedding"]
        elif "embeddings" in response:
            embeddings=response["embeddings"]
            if isinstance(embeddings[0],list):
                embedding=embeddings[0]
            else:
                embedding=embeddings
        else:
            print("chunk and vector adding failed")
            return 
        embedding_np=np.array(embedding,dtype="float")
        vector_DB.append((chunk,embedding_np))
        print(f"chunk and vector successfully added with shape{embedding_np.shape}")
    except Exception as e:
        print(f"{e}")
def cosine_similarity(a,b):
    a_np=np.array(a,dtype="float")
    b_np=np.array(b,dtype="float")
    dot_product=np.dot(a,b)
    norm_a=np.linalg.norm(a_np)
    norm_b=np.linalg.norm(b_np)
    if norm_a==0 or norm_b==0:
        return 0
    else:
        return dot_product/norm_a*norm_b

def retrive(query,top_n=3):
    try:
        responce=ollama.embed(model=embedded_model,input=query)
        if "embadding" in responce:
            embadding_query=responce["embadding"]
        elif "embaddings" in responce:
            embadding_query=responce["embaddings"][0]
        else:
            print("vector not returned")
            return
        embadded_query_np=np.array(embadding_query,dtype="float")
        similarity=[]
        for chunk,embedding in vector_DB:
            sim=cosine_similarity(embadded_query_np,embedding)
            similarity.append((chunk,sim))
        similarity.sort(key=lambda x:x[0],reverse=True)
        return similarity[:top_n]
    except Exception as e:
        print(f"error is {e}")
def main():
    print("loading dataset")
    with open("data.txt","r") as file1:
        dataset=file1.readlines
    dataset=[i.strip() for i in dataset]
    print(f"loaded{len(dataset)} entries")
    print("Indexing dataset(this might take few minute)....")
    for i,chunk in dataset:
        add_chuck_to_database(chunk)
        print(f'Added chunk {i+1}/ {len(dataset)} to the database')
    while True:
        query_input=("tell me about the luminar(or type EXiT to stop)")
        if query_input.lower()=="exit":
            break
        print("\nretriving Knowledge")
        retiving_knowledge=retrive(query_input)
        print("\nretivied knowledge")
        for chunk,similarity in retiving_knowledge:
            print(f"-{chunk}")
        instraction_prompt= f''' You are a helpful chatbot that knows about institue Luminar Technolab. Use only the following pieces of context to answer the question. 
        Dont make up any new information:{"\n".join([f'-{chunk}'for chunk,similarity in retiving_knowledge])}'''
        print("\nchatbot responce")
        s=ollama.chat(model=language_model,messages=
                      [{"role":"system","content":instraction_prompt},
                       {"role":"user","content":query_input}],stream=True)
        print("\nchatbot responce")
        for chunk in s:
            print(chunk["message"]["content"],end="",flush=True)
if __name__=="__main__":
    try:
        main()
    except Exception as e:
        print(f"\n error in {e}")