from rag4p.integrations.openai.openai_answer_generator import OpenaiAnswerGenerator
from rag4p.integrations.openai.openai_embedder import OpenAIEmbedder
from rag4p.integrations.weaviate.access_weaviate import AccessWeaviate
from rag4p.integrations.weaviate.weaviate_retriever import WeaviateRetriever
from rag4p.rag.retrieval.strategies.window_retrieval_strategy import WindowRetrievalStrategy
from rag4p.util.key_loader import KeyLoader

if __name__ == '__main__':
    """
    The goal for this step is to understand how it works to create an answer to a question given a context. The
    OpenAIAnswerGenerator is used to generate an answer. The OpenAIClient is used to communicate with the
    OpenAI API. The KeyLoader is used to load the OpenAI key from the environment variables. The LLM is also used
    to determine the quality of the answer. The answer is compared to the question, to see if it answers the question.
    The answer is also compared to the context, to see if it sticks to the context to prevent hallucination.
    """

    from dotenv import load_dotenv

    load_dotenv()

    key_loader = KeyLoader()
    embedder = OpenAIEmbedder(api_key=key_loader.get_openai_api_key())
    answer_generator = OpenaiAnswerGenerator(openai_api_key=key_loader.get_openai_api_key())

    ############################################################
    # Generate an answer
    ############################################################

    question = "What elements does a pipeline for q and a systems need?"
    context = ("We’ll be teaching you how to build an entire pipeline, starting from collecting data from various "
               "sources, converting that into vectors (yeah, it’s more math, but it’s cool, we promise), and storing "
               "it so you can use it to answer all sorts of queries.")

    # TODO 1: Generate an answer using the answer_generator. Use the question and context variables
    # begin solution

    # end solution

    ############################################################
    # Create the complete Q&A system
    ############################################################
    weaviate_client = AccessWeaviate(url=key_loader.get_weaviate_url(), access_key=key_loader.get_weaviate_api_key())
    weaviate_client.print_meta()

    openai_embedder = OpenAIEmbedder(api_key=key_loader.get_openai_api_key())
    weaviate_retriever = WeaviateRetriever(weaviate_access=weaviate_client,
                                           embedder=openai_embedder,
                                           additional_properties=["title", "time", "room", "speakers", "tags"],
                                           hybrid=False)

    question = "What is an alternative to keyword search?"
    strategy = WindowRetrievalStrategy(retriever=weaviate_retriever, window_size=1)
    result = strategy.retrieve_max_results(question=question, max_results=1)
    context = result.construct_context()

    # TODO 2: Generate an answer using the answer_generator, the obtained context and the question.
    #  Did it find an answer? If not, why not? What can you do?
    # begin solution

    # end solution

    # TODO 3: Play around with the different components, ask other questions, choose other strategy, etc

    weaviate_client.close()