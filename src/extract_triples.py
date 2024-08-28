from langchain.prompts import PromptTemplate
from langchain.llms.openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.text_splitter import CharacterTextSplitter, SpacyTextSplitter, NLTKTextSplitter
from langchain.graphs.networkx_graph import KG_TRIPLE_DELIMITER
from pprint import pprint
from pyvis.network import Network
import networkx as nx
import gradio as gr
from tqdm import tqdm

import nltk
nltk.download('punkt')

import openai
import json
openai.api_key = ('OPENAI_KEY')


def get_prompt(file_name):
    with open('models/prompt/'+file_name) as f: 
        dict = json.load(f)
    prompt = dict['prompt']
    return prompt

def get_report(file_name):
    with open("../data/"+file_name+".txt", 'r') as f:
        report = f.read().replace('\n', '')
    return report

def split_chunks(report):
    print("Splitting report into chunks")
    text_splitter = NLTKTextSplitter()
    docs = text_splitter.split_text(report)
    print("Number of chunks: ", len(docs))
    return docs

def parse_triples(response, delimiter=KG_TRIPLE_DELIMITER):
    if not response:
        return []
    return response.split(delimiter)

def to_dict(triples):
    triplets_json = []
    for triplet in triples:
        split = triplet.strip().split('~')
        if len(split) == 3:
            subject, predicate, obj = triplet.strip().split('~')
            triplets_json.append(
                {"subject": subject.strip(), "object": obj.strip(), "predicate": predicate.strip(), }
            )
    return triplets_json

def to_json(path, dict):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(dict, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    report_files = [
        "Deutsche_Bank_AG-Non-EN"
    ]

    prompt_file = "ontosustain_stso_2.json"

    prompt = get_prompt(prompt_file)

    KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT = PromptTemplate(
        input_variables=["text"],
        template=prompt,
    )

    print("Intialize chain")
    # Initialize chain
    # GPT 4: 8,192 tokens
    llm = ChatOpenAI(
        api_key=openai.api_key,
        temperature=0.9,
        model="gpt-4"
        )

    # Create an LLMChain using the knowledge triple extraction prompt
    chain = LLMChain(llm=llm, prompt=KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT)

    for report_file in report_files:
        print("Processing ", report_file)
        report = get_report(report_file)
        docs = split_chunks(report)

        all_triples = []
        print("Extracting triples ...")
        for doc in tqdm(docs):
            triples = chain.invoke(
                {'text' : doc}
            ).get('text')

            triples_data = parse_triples(triples)
            triples_list = triples_data[0].strip().split('|')
            triples_list_clean = list(filter(lambda x,m=max(map(len, triples_list)):len(x)>0, triples_list))
            all_triples.extend(triples_list_clean)

        print("Knowledge extracted")

        triples_dict = to_dict(all_triples)
        to_json("../results/raw/"+report_file+".json", triples_dict)
        print("Finished")

