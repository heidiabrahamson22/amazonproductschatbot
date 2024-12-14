from io import BytesIO
import pandas as pd
from langchain_chroma import Chroma
from langchain_experimental.open_clip import OpenCLIPEmbeddings
import os
import openai
import shutil
from operator import itemgetter
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI
from IPython.display import HTML, display
import base64
import io
import numpy as np
from PIL import Image
import streamlit as st
import pysqlite3

persist_directory = "/Users/Heidi/Downloads/chroma_data_storage"

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = Chroma(
        collection_name="multi_modal_store",
        persist_directory=persist_directory,
        embedding_function=OpenCLIPEmbeddings()
    )

vectorstore = st.session_state.vectorstore

retriever = vectorstore.as_retriever()

def resize_base64_image(base64_string, size=(128, 128)):
    """
    Resize an image encoded as a Base64 string.

    Args:
    base64_string (str): Base64 string of the original image.
    size (tuple): Desired size of the image as (width, height).

    Returns:
    str: Base64 string of the resized image.
    """
    # Decode the Base64 string
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))

    # Resize the image
    resized_img = img.resize(size, Image.LANCZOS)

    # Save the resized image to a bytes buffer
    buffered = io.BytesIO()
    resized_img.save(buffered, format=img.format)

    # Encode the resized image to Base64
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def is_base64(s):
    """Check if a string is Base64 encoded"""
    try:
        return base64.b64encode(base64.b64decode(s)) == s.encode()
    except Exception:
        return False


def split_image_text_types(docs):
    """Split numpy array images and texts"""
    images = []
    text = []
    metadata = []
    for doc in docs:
        doc = doc.page_content  # Extract Document contents
        if is_base64(doc):
            # Resize image to avoid OAI server error
            images.append(
                resize_base64_image(doc, size=(250, 250))
            )  # base64 encoded str
        else:
            text.append(doc)
                # Extract metadata (if available) and add to metadata list
        if hasattr(doc, 'metadata') and doc.metadata:
            metadata.append(doc.metadata)
        else:
            metadata.append({})  # Add an empty dictionary if no metadata
    return {"images": images, "texts": text,"metadata":metadata}

def prompt_func(data_dict, include_images=False):
    # Joining the context texts into a single string
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    messages = []

    if include_images and data_dict["context"]["images"]:
        # Adding image(s) to the messages if present
        first_image = data_dict["context"]["images"][0]
        image_message = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{first_image}"
            },
        }
        messages.append(image_message)

    # Adding the text message for analysis
    text_message = {
        "type": "text",
        "text": (
            "As a product analysis expert, your task is to evaluate and interpret both text descriptions and "
            "images from a dataset of Amazon products. You will be provided with product data retrieved from a "
            "vectorstore based on user-provided keywords or uploaded images. Please use your expertise to provide "
            "a detailed response tailored to the query type:\n\n"
            "For Text-Based Questions:\n"
            "- Summarize product features, specifications, and comparisons.\n"
            "- Provide clear and informative answers to help users understand product details or make decisions.\n"
            "- If comparing products, highlight similarities, differences, and key factors influencing the choice.\n\n"
            "For Image-Based Questions:\n"
            "- Identify the product in the image and describe its features and usage.\n"
            "- Analyze the image to align the visual elements with product categories and applications.\n"
            "- Include usage instructions if relevant.\n\n"
            "For Requests for Specific Product Images:\n"
            "- Search the dataset for matching images and provide them, if available.\n"
            "- Accompany the image with a brief description to confirm its relevance.\n\n"
            f"User-provided keywords: {data_dict['question']}\n\n"
            "Text descriptions:\n"
            f"{formatted_texts}"
        ),
    }
    messages.append(text_message)

    return [HumanMessage(content=messages)]

def should_include_images(question):
    # Simple logic to determine if the question requests images
    keywords = ["image", "photo", "picture", "visual"]
    return any(keyword in question.lower() for keyword in keywords)

st.title("Multimodal Product Analysis App")

st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

if not api_key:
    st.warning("Please enter your OpenAI API key in the sidebar to proceed.")
else:
    # Instantiate the OpenAI model using the provided API key
    model = ChatOpenAI(openai_api_key=api_key, temperature=0, model="gpt-4o", max_tokens=350)

    st.sidebar.header("Query Options")
    query = st.sidebar.text_input("Enter your query:", value="Describe a product")
    uploaded_image = st.sidebar.file_uploader("Upload an image (optional):", type=["jpg", "jpeg", "png"])

    if st.sidebar.button("Submit"):
        st.write("### Query Results")

        # Prepare input data
        if uploaded_image:
            # Load and save the uploaded image
            img = Image.open(uploaded_image)
            image_path = f"uploaded_image_{uploaded_image.name}"  # Save with its original name
            img.save(image_path, format="JPEG")  # Save image to the path

            # Add the image path to the query
            query = f"{query}. The related image is located at: {image_path}"

        # Define RAG chain
        chain = (
            {
                "context": retriever | RunnableLambda(split_image_text_types),
                "question": RunnablePassthrough(),
            }
            | RunnableLambda(
                lambda data: {
                    "prompt": prompt_func(data, include_images=should_include_images(data["question"])),
                    "images": [data["context"]["images"][0]] if should_include_images(data["question"]) and data["context"]["images"] else [],
                }
            )
            | RunnableLambda(
                lambda data: {
                    "text": model.invoke(data["prompt"]),
                    "images": data["images"],
                }
            )
        )

        # Run the chain
        result = chain.invoke(query)

        # Display text result
        st.write("#### Text Output:")
        st.write(result["text"].content)

        # Display images (if any)
        if result.get("images"):
            st.write("#### Images:")
            for img_base64 in result["images"]:
                img_data = BytesIO(base64.b64decode(img_base64))
                img = Image.open(img_data)
                st.image(img, use_container_width=True)
