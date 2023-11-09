from flask import Flask, request, jsonify
import langchain
from langchain.llms import OpenAI
from langchain import PromptTemplate, LLMChain
from flask_cors import CORS, cross_origin

import re
import os
import pandas as pd
import bs4
import requests
import spacy
from spacy import displacy

# Load the English language model
nlp = spacy.load("en_core_web_sm")

from spacy.matcher import Matcher
from spacy.tokens import Span

import networkx as nx

app = Flask(__name__)
cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"

text_file = r"E:\Work\workbackups\localdata\apikey.txt"

with open(text_file, "r") as f:
    api_key = f.read().strip()

llm = OpenAI(openai_api_key=api_key)

# import wikipedia sentences
candidate_sentences = pd.read_csv(
    "E:\Work\workbackups\llm\sentencetrial2.csv"
)

df = pd.DataFrame(
    {
        "source": list(
            i.strip() for i in candidate_sentences['source']
        ),
        "edge": list(
            i.strip() for i in candidate_sentences['edge']
        ),
        "target": list(
            i.strip() for i in candidate_sentences['target']
        ),
    }
)

# create a directed-graph from a dataframe
G = nx.from_pandas_edgelist(
    df, "source", "target", edge_attr=True, create_using=nx.MultiDiGraph()
)

conversation_history = []


def return_context(noun):
    x = noun

    sentences = []

    if x in G.nodes:
        neighbor_nodes = list(G.neighbors(x))  # get the list of neighbors of node n
        neighbor_nodes.append(x)  # add node n to the list
        H = G.subgraph(neighbor_nodes)

        neighbor_edges = list(G.edges(x, data=True))

        for u, v, d in neighbor_edges:
            verb = d["edge"]  # get the verb from the edge label
            sentences.append(
                f"{u} {verb} {v}."
            )  # create a sentence with the source, verb,

    return sentences


def get_entities(sentence):
    # Parse the sentence using spaCy
    doc = nlp(sentence)

    # Print the text and label of each noun phrase
    for chunk in doc.noun_chunks:
        print(chunk.text, chunk.label_)

    lister = [i.text for i in doc.noun_chunks]

    for i in lister:
        lisst = i.split(" ")

        if "the" in lisst:
            lisst.remove("the")

        sentence = ""
        for x in lisst:
            sentence = sentence + str(x)

        i = x
    return lister


@app.route("/ask", methods=["POST"])
@cross_origin()
def ask():
    global conversation_history
    question = request.json["question"]

    entities = get_entities(question)

    context = []
    for i in entities:
        context.append(return_context(i))

    # for j in context:
    #     question = question + j

    conversation_history.append(str(context))

    prompt = PromptTemplate(
        template=" ".join(conversation_history) + "\nQ: {question}\n A: ",
        input_variables=["question"],
    )

    llm_chain = LLMChain(prompt=prompt, llm=llm)
    response = llm_chain.run(question)
    return jsonify(
        {
            "response": response.replace("\n", ""),
            "context": context,
            "entities": entities,
            "history": conversation_history,
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)
