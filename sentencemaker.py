import bs4
import requests
import spacy
from spacy import displacy

# Load the English language model
nlp = spacy.load("en_core_web_sm")

from spacy.matcher import Matcher
from spacy.tokens import Span

import networkx as nx
import pandas as pd

from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

model = "lmsys/vicuna-7b-v1.1"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)

candidate_sentences = pd.read_csv(
    "/var/wd_smit/external_repos/knowledge-graph/data/output/kg/input_data.txt-out.csv"
)

df = pd.DataFrame(
    {
        "source": list(
            i.strip() for i in candidate_sentences[candidate_sentences.columns[0]]
        ),
        "edge": list(
            i.strip() for i in candidate_sentences[candidate_sentences.columns[1]]
        ),
        "target": list(
            i.strip() for i in candidate_sentences[candidate_sentences.columns[2]]
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

        i = sentence
    return lister


question = "Who has rights to The Godfather?"
entities = get_entities(question)

context = []
for i in entities:
    context.append(return_context(i))

lmax = 0
for j in context:
    if len(j) != 0:
        for k in j:
            if len(k) >= lmax:
                lmax = len(k)
        for k in j:
            if len(k) == lmax:
                question = question + " Context: "
                question = question + str(k)


# question = question + " Answer: "

sequences = pipeline(
    question,
    max_length=100,
    do_sample=True,
    top_k=50,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    temperature=0.3,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")

print("--------------------------------")
print("Question: " + question)

print("--------------------------------")
print("Contexts: " + str(context))

print("--------------------------------")
print("Nodes: ")
print(list(G.nodes))
