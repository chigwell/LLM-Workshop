# Plan
# 1. The basic usage
# 2. Embeddings
# 3. Structure
# 4. CV
# 5. Function calling
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


llm = ChatOllama(model="llama3.1:8b", temperature=0, stream=True)

human_message = HumanMessage(
    content="What is the capital of France?",
)


texts = ["""
Benchmarking Multimodal Models for Fine-Grained
Image Analysis: A Comparative Study Across Diverse
Visual Features.
Evgenii Evstafev A
A University Information Services (UIS), University of Cambridge,
Roger Needham Building, 7 JJ Thomson Ave, Cambridge CB3 0RB, UK, ee345@cam.ac.uk
ABSTRACT
This article introduces a benchmark designed to evaluate the capabilities of multimodal models in analyzing and
interpreting images. The benchmark focuses on seven key visual aspects: main object, additional objects, background, detail, dominant colors, style, and viewpoint. A dataset of 14,580 images, generated from diverse text
prompts, was used to assess the performance of seven leading multimodal models. These models were evaluated
on their ability to accurately identify and describe each visual aspect, providing insights into their strengths and
weaknesses for comprehensive image understanding. The findings of this benchmark have significant implications
for the development and selection of multimodal models for various image analysis tasks.
TYPE OF PAPER AND KEYWORDS
Benchmarking, multimodal models, image analysis, computer vision, deep learning, image understanding, visual
features, model evaluation
1 INTRODUCTION
Multimodal models, capable of processing and integrating information from multiple modalities such as
text and images [1], have emerged as a powerful tool for
comprehensive image understanding [2]. These models
hold the potential to revolutionize various applications,
including image retrieval, content creation, and humancomputer interaction. However, evaluating their ability
to capture fine-grained details and contextual information remains a crucial challenge [3]. This article presents a benchmark for evaluating the performance of different multimodal models in identifying and analyzing
specific aspects of images, such as the main object, additional objects, background, details, dominant colors,
style, and viewpoint. By comparing their performance
across a range of tasks, this research aims to provide insights into the strengths and weaknesses of different
multimodal approaches for fine-grained image analysis.
2. BACKGROUND AND RELATED WORK
Multimodal models in computer vision use the interplay between different modalities, such as text and images, to achieve a more holistic understanding of visual
content [4]. This approach has shown promising results
in various tasks, including image captioning, visual
question answering, and image generation [5]. Recent
advancements in deep learning, particularly the development of transformer-based architectures, have further
accelerated progress in this field. Models like CLIP
(Contrastive Language-Image Pre-training [6]) have
demonstrated the ability to learn robust representations
that capture the semantic relationship between images
and text [7]. However, there is a need for standardized
benchmarks to evaluate the performance of multimodal
models in fine-grained image analysis, as existing
benchmarks often focus on broader tasks without explicitly assessing their ability to capture subtle details and
contextual information [8]. This research addresses this
gap by introducing a benchmark that specifically targets
the analysis of diverse visual features in images, enabling a more comprehensive evaluation of their capabilities.
3. METHODOLOGY
3.1 DATASET CREATION
The dataset creation process involved generating a
diverse set of image descriptions (prompts) by systematically
"""
]

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

embeddings = HuggingFaceEmbeddings()
vectorestore = FAISS.from_texts(texts, embeddings)
retriever = vectorestore.as_retriever()

prompt = ChatPromptTemplate.from_template(
    "Given the context: {context}, answer the folowing question: {query}"
)

chain = (
    {"context": retriever, "query": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

response = chain.invoke("Сколько изображений было использовано для оценки производительности семи ведущих мультимодальных моделей?")
print(response)


from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


class ImageCount(BaseModel):
    number: int = Field(..., title="The number of images used for evaluation", description="The number of images used for evaluation")

parser = PydanticOutputParser(pydantic_object=ImageCount)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's query based on the context below. Ensure you follow the format instructions:\n{format_instructions}"),
    ("human", "Question: {query}\nContext: {context}")
]).partial(format_instructions=parser.get_format_instructions())


chain = (
    {"context": retriever, "query": RunnablePassthrough()}
    | prompt_template
    | llm
    | parser
)

result = chain.invoke("How many images were used to evaluate the performance of seven leading multimodal models?")
print(result.number)


import base64
from PIL import Image
import io

def image_to_base64(image_path):
    with Image.open(image_path) as img:
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Create a byte buffer
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")

        # Encode to base64
        return base64.b64encode(buffer.getvalue()).decode('utf-8')


class ArxivID(BaseModel):
    arxiv_id: str = Field(..., title="ArXiv ID", description="The ArXiv ID of the article")

parser = PydanticOutputParser(pydantic_object=ArxivID)

image_path = "test.png"

llm = ChatOllama(model="minicpm-v:8b", temperature=0)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's query based on the context below. Ensure you follow the format instructions:\n{format_instructions}"),
    ("human", "Question: {query}"),
    HumanMessage(
        content=[
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_to_base64(image_path)}"
                }
            }
        ]

    )
]).partial(format_instructions=parser.get_format_instructions())

chain = (
    {"context": retriever, "query": RunnablePassthrough()}
    | prompt_template
    | llm
    | parser
)
result = chain.invoke("What is the ArXiv ID of the article?")
print(result.arxiv_id)

from langchain_core.tools import tool

@tool
def location_search(city: str) -> dict:
    """Search for a city in locations.txt file and return its postcode """
    try:
        print("Searching for city")
        with open("locations.txt", "r") as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split(",")
                print(parts[0].lower())
                if len(parts) >=3 and city.lower() == parts[0].lower():
                    return {
                        "city": parts[0],
                        "country": parts[1],
                        "postcode": parts[2]
                    }
            return {"error": f"City {city} not found"}
    except Exception as e:
        return {"error": str(e)}


llm = ChatOllama(model="llama3.1:8b", temperature=0).bind_tools([location_search])

prompt = ChatPromptTemplate.from_template(
    """Answer the user's question using tool available. Query: {query}
    Always use the location_search tool for city postcode queries!"""
)

def handle_tool_calls(output):
    if not isinstance(output, list):
        output = [output]
    for message in output:
        if hasattr(message, "tool_calls"):
            for tool_call in message.tool_calls:
                result = location_search(tool_call['args'])
                return f"The postcode for {tool_call['args']['city']} is {result['postcode']}"
    return "No postcode found"

chain = (
    RunnablePassthrough()
    | prompt
    | llm
    | handle_tool_calls
)
query = "Search for the postcode of Cambridge"
result = chain.invoke({"query": query})
print(result)

