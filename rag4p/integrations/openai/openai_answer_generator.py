from openai import OpenAI

from rag4p.integrations.openai import DEFAULT_MODEL
from rag4p.rag.generation.answer_generator import AnswerGenerator


class OpenaiAnswerGenerator(AnswerGenerator):
    """
    Used to generate answers to questions using the OpenAI API.
    """
    def __init__(self, openai_api_key: str, openai_model: str = DEFAULT_MODEL):
        self.openai_client = OpenAI(
            api_key=openai_api_key,
        )
        self.openai_model = openai_model

    def generate_answer(self, question: str, context: str) -> str:
        completion = self.openai_client.chat.completions.create(
            model=self.openai_model,
            messages=[
                {"role": "system", "content": "You are an assistant answering questions using the context provided. "
                                              "If the context does not contain the answer, you should tell you cannot "
                                              "answer using the context. The question is provided after 'question:'. "
                                              "The context after 'context:'."},
                {"role": "user", "content": f"Context: {context}\nQuestion: {question}\nAnswer:"},
            ],
            stream=False,
        )

        return completion.choices[0].message.content
