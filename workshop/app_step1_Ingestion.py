from rag4p.indexing.splitters.sentence_splitter import SentenceSplitter
from rag4p.indexing.splitters.max_token_splitter import MaxTokenSplitter
from rag4p.indexing.input_document import InputDocument
from rag4p.rag.store.local.internal_content_store import InternalContentStore
from rag4p.rag.embedding.local.onnx_embedder import OnnxEmbedder
from rag4p.integrations.openai.openai_embedder import OpenAIEmbedder
from rag4p.util.key_loader import KeyLoader

from workshop.embedding.alphabet_embedder import AlphabetEmbedder

if __name__ == '__main__':
    """
    The goal for step 1 is to understand the required elements for ingesting content into a vector database:
    - 1. Read the content from a source
    - 2. Split the content into chunks using a splitter.
    - 3. Create a vector or embedding from the text using an embedder. 
    
    The size of the chunks you create by the splitter has a big impact on the precision of the system. If 
    the chunks are too small, the system will have a hard time to find the correct answer. If the chunks are too 
    big, the system will have a hard time to find the correct answer. The goal is to find a good balance between 
    the two.
    
    The embedders are essential to the similarity search of the vector store. You will use a weird embedder
    called the AlphabetEmbedder. This embedder will create an embedding for a chunk of text based on the alphabet.
    A better, still local running, embedder is the OnnxEmbedder. The best embedder you use is the OpenAIEmbedder.
    """

    from dotenv import load_dotenv
    load_dotenv()

    key_loader = KeyLoader()

    # Gathering the content
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

    ############################################################
    # Splitting the content
    ############################################################
    num_chunks = 0
    splitter = None
    # TODO 1: Use the sentence splitter to split content into chunks.
    # BEGIN SOLUTION
    splitter = SentenceSplitter()
    chunks = splitter.split(doc)
    num_chunks = len(chunks)
    # END

    print(f"Number of chunks is : {'correct' if num_chunks == 16 else 'wrong'} ")

    # TODO 2: Print the text of the chunks to verify the sentences. We think 17 would be better.
    # Spot the problem in the content that causes the splitter to create 16 chunks.
    # BEGIN SOLUTION
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1}: {chunk.chunk_text}")
    # END

    # sys.exit(0)
    ############################################################
    # Storing content embeddings
    ############################################################

    embedder = OpenAIEmbedder(api_key=KeyLoader().get_openai_api_key())  # TODO copy the solution from this embedder
    embedding = embedder.embed(chunks[0].chunk_text)
    # TODO 3: Look at the embedding. What is the length of the embedding? How does the embedding work?
    # BEGIN SOLUTION
    print(f"Embedding: {embedding}")
    print(f"Embedding size: {len(embedding)}")
    # END

    content_store = InternalContentStore(embedder=embedder)
    content_store.store(chunks=chunks)
    relevant_chunks = content_store.find_relevant_chunks(query="What is the future after traditional searches?")

    # TODO 4: Print the relevant chunks. What is your opinion on the results? Did you print the score?
    # BEGIN SOLUTION
    for chunk in relevant_chunks:
        print(f"Document: {chunk.document_id}")
        print(f"Chunk id: {chunk.chunk_id}")
        print(f"Text: {chunk.chunk_text}")
        print(f"Score: {chunk.score:.3f}")
        print("--------------------------------------------------")
    # END

    # TODO 5: Replace the embedder with the OnnxEmbedder. What is the size of the embedding Now? Do the results improve?
    # embedder = OnnxEmbedder()

    # TODO 6: Replace the embedder with the OpenAIEmbedder. What is the size of the embedding Now? Do the results improve?
    # embedder = OpenAIEmbedder(api_key=key_loader.get_openai_api_key())

    # TODO 7: Replace the splitter by a MaxTokenSplitter. What is the number of chunks now? What is the
    #  impact on the results?
    # splitter = MaxTokenSplitter(max_tokens=50)
