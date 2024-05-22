from rag4p.integrations.openai.openai_answer_generator import OpenaiAnswerGenerator
from rag4p.rag.generation.observed_answer_generator import ObservedAnswerGenerator
from rag4p.rag.tracker.rag_tracker import global_data
from rag4p.rag.retrieval.strategies.window_retrieval_strategy import WindowRetrievalStrategy
from workshop.retrieval.strategies.document_retrieval_strategy import DocumentRetrievalStrategy
from rag4p.util.key_loader import KeyLoader
from rag4p.integrations.openai.quality.openai_answer_quality_service import OpenAIAnswerQualityService
from rag4p.integrations.openai.openai_embedder import OpenAIEmbedder
from rag4p.integrations.weaviate.access_weaviate import AccessWeaviate
from rag4p.integrations.weaviate.weaviate_retriever import WeaviateRetriever

if __name__ == '__main__':
    """
    This script is a bonus step that shows how to determine the quality of your RAG system. The quality is depending on
    the quality of all components. First we have the retriever that finds the relevant items for the question. The
    retriever strategy constructs the context. The generator is responsible for answering the question. If the generated
    answer is no answer to the question, the quality is low. If the generator creates a good answer to the question, we
    still need to verify if the answer is related to the context. If the answer is not related to the context, the 
    quality is low. Therefore we have three quality metrics:
    - Retrieval quality
    - Answer to question quality
    - Answer from context quality

    In this script you explore all three quality metrics.
    """

    from dotenv import load_dotenv

    load_dotenv()

    key_loader = KeyLoader()

    ############################################################
    # Answer to Question quality + Answer from Context quality
    ############################################################
    """
    To determine the quality of the answer, you use the LLM model from OpenAI. We prompt the LLM to rate the answer
    with a number between 1 and 5, where 1 is the worst and 5 is the best. At first we present the question and the 
    answer.
    """

    question = "What session does this text describe?"

    context = ("Ever thought about building your very own question-answering system? Like the one that powers Siri, "
               "Alexa, or Google Assistant? Well, we've got something awesome lined up for you! In our hands-on "
               "workshop, we'll guide you through the ins and outs of creating a question-answering system. You can "
               "use Python or Java for the workshop. You'll get your hands dirty with vector stores and Large Language "
               "Models, we help you combine these two in a "
               "way you've never done before. You've probably used search engines for keyword-based searches, "
               "right? Well, prepare to have your mind blown. We'll dive into something called semantic search, "
               "which is the next big thing after traditional searches. It’s like moving from asking Google to search "
               "\"best pizza places\" to \"Where can I find a pizza place that my gluten-intolerant, vegan friend "
               "would love?\" you get the idea, right? We’ll be teaching you how to build an entire pipeline, "
               "starting from collecting data from various sources, converting that into vectors (yeah, it’s more math,"
               " but it’s cool, we promise), and storing it so you can use it to answer all sorts of queries. It's like"
               " building your own mini Google! We've got a repository ready to help you set up everything you need on "
               "your laptop. By the end of our workshop, you'll have your question-answering system ready and "
               "running.So, why wait? Grab your laptop, bring your coding hat, and let's start building something "
               "fantastic together.")

    openai_answer_generator = OpenaiAnswerGenerator(openai_api_key=key_loader.get_openai_api_key())
    answer_generator = ObservedAnswerGenerator(answer_generator=openai_answer_generator)
    answer = answer_generator.generate_answer(question, context)

    rag_observer = global_data["observer"]

    print(f"Question: {rag_observer.question}")
    print(f"Context: {rag_observer.context}")
    print(f"Answer: {answer}")

    answer_quality_service = OpenAIAnswerQualityService(openai_api_key=key_loader.get_openai_api_key(),
                                                        openai_model="gpt-4o")
    quality = answer_quality_service.determine_quality_answer_related_to_question(rag_observer=rag_observer)
    print(f"Quality: {quality.quality}, Reason: {quality.reason}")
    quality = answer_quality_service.determine_quality_answer_from_context(rag_observer=rag_observer)
    print(f"Quality: {quality.quality}, Reason: {quality.reason}")

    rag_observer.reset()

    # TODO 1: Play around with the context and the question.

    # TODO 2: If you feel advanturous, try to create the full application where the context is obtain through a
    #  retrieval strategy and the answer is generated by the generator. Then determine the quality of the answer.
    embedder = OpenAIEmbedder(api_key=key_loader.get_openai_api_key())
    weaviate_client = AccessWeaviate(url=key_loader.get_weaviate_url(),
                                     access_key=key_loader.get_weaviate_api_key())
    retriever = WeaviateRetriever(weaviate_access=weaviate_client,
                                  embedder=embedder,
                                  hybrid=True,
                                  additional_properties=["title", "time", "room", "speakers", "tags"])

    question = "Who are the people presenting the workshop about a q&a system?"
    print("\n----------------------------------")
    # BEGIN SOLUTION

    # END
    weaviate_client.close()
