# chatbot using local vector DB
# and remote model=OpenAI GPT-3
'''
After python environment setup, ensure that these packages are installed:
pip install langchain langchain-openai tiktoken # for openai model access
pip install faiss-cpu # for local vector DB
pip install beautifulsoup4 # for web scraping

To use local vector DB, ensure that Ollama server is running locally
we will use llama2 as embedding model and FAISS as vector DB from Meta
'''
import os
import os
import json


def app_setup():
    # replace values for your specific api keys in the config.json file.
    # we load and read the config
    home_dir = os.path.expanduser("~")
    cfgFile = os.path.join(home_dir, ".langchain", "config.json")
    configData = json.load(open(cfgFile, "r"))

    # read and set all environment variables
    os.environ["OPENAI_API_KEY"] = configData["openAI"]["apiKey"]
    os.environ["OLLAMA_HOST"] = "127.0.0.1" # we are running the ollama server locally
    # openai_api_key = os.getenv("OPENAI_API_KEY")
    
    
def run_chatbot():
    print("Hello, I am ChatGPT 3.5 Turbo. I am a chatbot.")
    # now initialize the OpenAI Chat model:
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI()
    print(llm.model_name)
    llm.invoke("how can langsmith help with testing?")
    from langchain_core.prompts import ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are world class technical documentation writer."),
        ("user", "{input}")
    ])
    
    chain = prompt | llm # combine chains
    chain.invoke({"input": "how can langsmith help with testing?"})
    from langchain_core.output_parsers import StrOutputParser
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser # combine chains
    chain.invoke({"input": "how can langsmith help with testing?"})
    # we will populate a vector store and use that as a retriever
    # use the WebBaseLoader to load some web data and then vectorize it
    # now use webBaseLoader
    from langchain_community.document_loaders import WebBaseLoader
    loader = WebBaseLoader("https://docs.smith.langchain.com/overview")
    docs = loader.load()
    '''
    Next, we need to index it into a vectorstore. 
    This requires a few components, namely an embedding model and a vectorstore.
    For embedding models, we can use OpenAI or local models.
    For local, ensure you have Ollama running (same set up as with the LLM).
    '''
    
    '''
    # for OpanAI embeddings we can use:
    # from langchain_openai import OpenAIEmbeddings
    # embeddings = OpenAIEmbeddings()
    # We'll need a local instance of Ollam running. Ensur ethis is running first before invoking calls to the embeddings
    # we can use a local docker image, to start a locall instance of Ollam in docker:
    # for docker container use:
    docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
    '''
    # for now we will use Ollama embeddings running locally
    from langchain_community.embeddings import OllamaEmbeddings
    embeddings = OllamaEmbeddings()

    '''
    Now, we can use this embedding model to ingest documents into a vectorstore. \
    We will use a simple local vectorstore, FAISS, for simplicity's sake.\
    First we need to install the required packages for that:
    '''
    # Now build the index:
    from langchain_community.vectorstores import FAISS
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(docs)
    vector = FAISS.from_documents(documents, embeddings)
    
    '''
    Now that we have this data indexed in a vectorstore, we will create a retrieval chain. \
    This chain will take an incoming question, look up relevant documents, \
    then pass those documents along with the original question into an LLM and ask \
    it to answer the original question.
    First, let's set up the chain that takes a question and the \
    retrieved documents and generates an answer.
    '''
    from langchain.chains.combine_documents import create_stuff_documents_chain
    prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

    <context>
    {context}
    </context>
    Question: {input}""")
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    '''
    we want the documents to first come from the retriever we just set up. \
    That way, for a given question we can use the retriever to dynamically \
    select the most relevant documents and pass those in.
    '''
    from langchain.chains import create_retrieval_chain
    retriever = vector.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    '''
    We can now invoke this retrieval chain. \
    This returns a dictionary - the response from the LLM is in the answer key
    '''
    response = retrieval_chain.invoke({"input": "how can langsmith help with testing?"})
    print(response["answer"])
    
    # Now update the code to create a Conversation Retrieval Chain
    # In order to update retrieval, we will create a new chain. 
    # This chain will take in the most recent input (input) and the 
    # conversation history (chat_history) and use an LLM to generate a search query.

    from langchain.chains import create_history_aware_retriever
    from langchain_core.prompts import MessagesPlaceholder
    # First we need a prompt that we can pass into an LLM to generate this search query
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    # We can test this out by passing in an instance where the user is asking a follow up question.
    from langchain_core.messages import HumanMessage, AIMessage

    chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]
    retriever_chain.invoke({
        "chat_history": chat_history,
        "input": "Tell me how"
    })
    
    '''
    You should see that this returns documents about testing in LangSmith. \
    This is because the LLM generated a new query, combining \
    the chat history with the follow up question.

    Now that we have this new retriever, we can create a new \
    chain to continue the conversation with these retrieved documents in mind.
    '''
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)

    # We can now test this out end-to-end:
    chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]
    retrieval_chain.invoke({
        "chat_history": chat_history,
        "input": "Tell me how"
    })
    
    '''
    We can see that this gives a coherent answer. Successfully completed chatbot!
    '''  
    print("We've successfully turned our retrieval chain into a chatbot!")

    
def main():
    app_setup()
    run_chatbot()

if __name__ == "__main__":
    main()