import sys
from rag4p.integrations.openai.openai_embedder import OpenAIEmbedder
from rag4p.integrations.weaviate.access_weaviate import AccessWeaviate
from rag4p.integrations.weaviate.weaviate_retriever import WeaviateRetriever
from rag4p.util.key_loader import KeyLoader
from rag4p.indexing.input_document import InputDocument
from rag4p.indexing.splitters.sentence_splitter import SentenceSplitter
from rag4p.rag.embedding.local.onnx_embedder import OnnxEmbedder
from rag4p.rag.store.local.internal_content_store import InternalContentStore
from rag4p.rag.retrieval.strategies.topn_retrieval_strategy import TopNRetrievalStrategy
from rag4p.rag.retrieval.strategies.window_retrieval_strategy import WindowRetrievalStrategy

if __name__ == '__main__':
    """
    The goal for step 2 is to understand how to retrieve content from a vector database. You used the retriever in
    step 1 to find matching chunks. In this step we take it a step further. In this step you learn about retrieval
    strategies. We use the retriever against a local vector database, and we use it to query Weaviate. Weaviate 
    contains embeddings of all the sentences using the default OpenAIEmbedder.
    """
    from dotenv import load_dotenv
    load_dotenv()

    key_loader = KeyLoader()

    ############################################################
    # Retrieval strategies
    ############################################################
    source = ("Ever thought about building your very own question-answering system? Like the one that powers Siri, "
              "Alexa, or Google Assistant? Well, we've got something awesome lined up for you! In our hands-on "
              "workshop, we'll guide you through the ins and outs of creating a question-answering system. You can use "
              "Python or Java for the workshop. You'll get your hands dirty with vector stores and Large Language "
              "Models, we help you combine these two in a "
              "way you've never done before. You've probably used search engines for keyword-based searches, "
              "right? Well, prepare to have your mind blown. We'll dive into something called semantic search, "
              "which is the next big thing after traditional searches. It’s like moving from asking Google to search "
              "\"best pizza places\" to \"Where can I find a pizza place that my gluten-intolerant, vegan friend "
              "would love?\" you get the idea, right? We’ll be teaching you how to build an entire pipeline, "
              "starting from collecting data from various sources, converting that into vectors (yeah, it’s more math, "
              "but it’s cool, we promise), and storing it so you can use it to answer all sorts of queries. It's like "
              "building your own mini Google! We've got a repository ready to help you set up everything you need on "
              "your laptop. By the end of our workshop, you'll have your question-answering system ready and "
              "running.So, why wait? Grab your laptop, bring your coding hat, and let's start building something "
              "fantastic together.")
    doc = InputDocument(
        document_id="jettro-daniel-teqnation-workshop-2022-03-30",
        text=source,
        properties={}
    )
    splitter = SentenceSplitter()
    embedder = OnnxEmbedder()

    content_store = InternalContentStore(embedder=embedder)
    content_store.store(chunks=splitter.split(doc))
    print("--------------------------------------------------")

    strategy = TopNRetrievalStrategy(retriever=content_store)
    # TODO 1: Retrieve only the best matching chunk for the query "What is an alternative to keyword search?"
    #  Can you answer the question based on the retrieved chunk? (max_results=1)
    # BEGIN SOLUTION

    # END

    # TODO 2: Use the window retrieval strategy, what size of the window would you use to answer the question?
    # BEGIN SOLUTION

    # END

    # sys.exit(0)
    ############################################################
    # Retrieve content from Weaviate
    ############################################################
    client = AccessWeaviate(url=key_loader.get_weaviate_url(), access_key=key_loader.get_weaviate_api_key())
    client.print_meta()

    openai_embedder = OpenAIEmbedder(api_key=key_loader.get_openai_api_key())
    weaviate_retriever = WeaviateRetriever(weaviate_access=client,
                                           embedder=openai_embedder,
                                           additional_properties=["title", "time", "room", "speakers", "tags"],
                                           hybrid=False)

    # TODO 3: Retrieve the first chunk of the document with the id "using-ai-to-save-time-and-sanity"
    #  How many chunks does the document have?
    # BEGIN SOLUTION

    # END

    # TODO 4: Retrieve the four best matching chunks for the query "Wie heeft het over zoek technologie?",
    #  use the window retrieval strategy with a window size of 2.
    query = "Wie heeft het over zoek technologie?"
    # BEGIN SOLUTION

    # END
    client.close()