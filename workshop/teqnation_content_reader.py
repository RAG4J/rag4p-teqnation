import json

from rag4p.indexing.content_reader import ContentReader
from rag4p.indexing.input_document import InputDocument


class TeqnationContentReader(ContentReader):
    def __init__(self):
        self.filename = "../data/teqnation/sessions.jsonl"

    def read(self):
        with open(self.filename, 'r') as file:
            for line in file:
                data = json.loads(line)
                properties = {
                    "speakers": data["speakers"],
                    "title": data["title"],
                    "room": data["room"],
                    "time": data["time"],
                    "tags": data["tags"]
                }
                document_id = data["title"].lower().replace(" ", "-")
                document = InputDocument(
                    document_id=document_id,
                    text=data["description"] if "description" in data else "",
                    properties=properties
                )
                yield document
