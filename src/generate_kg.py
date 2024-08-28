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

import nltk
nltk.download('punkt')

import openai
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import re

openai.api_key = ('OPENAI_KEY')


def read_json(path):
    with open(path, 'r') as f:
        content = json.load(f)
    return content

def to_json(path, dict):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(dict, f, ensure_ascii=False, indent=4)

def most_common(lst):
    return max(set(lst), key=lst.count)

def truncate_prompt(prompt: str, max_tokens: int) -> str:
    tokens = prompt.split()  # simple whitespace tokenization
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return ' '.join(tokens)

def get_ts_descriptions(content):
    ts_sentences = {
        "Impact": [],
        "Risks": [],
        "Opportunities": [],
        "Strategy": [],
        "Actions": [],
        "Targets": [],
        "Organization": [],
        "Reporting": [],
        "Reporting_date": []
        }

    for c in content:
        if c['object'] != "N/A":
            if c['subject'] == "Organization":
                ts_sentences['Organization'].append(c['object'])
            if c['subject'] == "Reporting":
                if c['predicate'] == "hasName":
                    ts_sentences['Reporting'].append(c['object'])
                if c['predicate'] == "releaseDate":
                    ts_sentences['Reporting_date'].append(c['object'])
            else:
                if len(c['object']) > 50:
                    if c['subject'] == "Impact":
                        ts_sentences['Impact'].extend(c['object'].strip().split('\n'))
                    if c['subject'] == "Risks":
                        ts_sentences['Risks'].extend(c['object'].strip().split('\n'))
                    if c['subject'] == "Opportunities":
                        ts_sentences['Opportunities'].extend(c['object'].strip().split('\n'))
                    if c['subject'] == "Strategy" and c['predicate'] == 'hasDescription':
                        ts_sentences['Strategy'].extend(c['object'].strip().split('\n'))
                    if c['subject'] == "Actions":
                        ts_sentences['Actions'].extend(c['object'].strip().split('\n'))
                    if c['subject'] == "Targets" :
                        ts_sentences['Targets'].extend(c['object'].strip().split('\n'))
    
    return ts_sentences

def generate_ts_prompts():
    N_SENTENCES = 2
    
    ts_prompts = {
        "Impact": f"Summurise negative impact on climate change company addresses in {N_SENTENCES} sentences.",
        "Risks": f"Summurise material risks from company's impact on climate change in {N_SENTENCES} sentences.",
        "Opportunities": f"Summurise financial materiality and the effect of climate change on the company in {N_SENTENCES} sentences.",
        "Strategy": f"Summurise company's strategy and business model in line with the transition to a sustainable economy in {N_SENTENCES} sentences.",
        "Actions": f"Summurise actions and resources in relation to material sustainability matters in {N_SENTENCES} sentences.",
        "Targets": f"Summurise company's goals towards sustainable economy in {N_SENTENCES} sentences."
    }
    
    return ts_prompts

if __name__ == "__main__":
    report_files = [
        "Deutsche_Bank_AG-Non-EN"
    ]

    print("Initialize chain")
    # Initialize chain
    # GPT 4: 8,192 tokens
    llm = ChatOpenAI(
        api_key=openai.api_key,
        temperature=0.9,
        model="gpt-4",
        # max_tokens=11000
        )
    
    for report_file in report_files:
        print("Processing ", report_file)
        raw_json_path = "../results/raw/"+report_file+".json"
        raw_content = read_json(raw_json_path)

        ts_descriptions = get_ts_descriptions(raw_content)
        ts_prompts = generate_ts_prompts()
        start_prompt = "Given list of sentences. "
        end_prompt = "Add line breaks between output sentences. \n Sentences: {text}"
        print("Topical Standard prompts generated")

        out_ts = {
            "Impact": "",
            "Risks": "",
            "Opportunities": "",
            "Strategy": "",
            "Actions": "",
            "Targets": ""
        }

        print("Summurizing standard descriptions ...")
        for aspect in tqdm(ts_prompts.keys()):
            template_prompt = PromptTemplate(
                input_variables=["text"], template=start_prompt+ts_prompts[aspect]+end_prompt
            )
            chain = LLMChain(llm=llm, prompt=template_prompt)

            out = chain.invoke(
                {'text' : truncate_prompt('.'.join(ts_descriptions[aspect]), 1500)}
            ).get('text')

            out_ts[aspect] = out

        processed_triples_dict = read_json("models/prompt/ontology_structure.json")
        processed_triples_dict.append(
            {
                "subject": "Organization",
                "object": most_common(ts_descriptions['Organization']),
                "predicate": "hasName"
            }
        )
        processed_triples_dict.append(
            {
                "subject": "Reporting",
                "object": most_common(ts_descriptions['Reporting']),
                "predicate": "hasName"
            }
        )
        processed_triples_dict.append(
            {
                "subject": "Reporting",
                "object": most_common(ts_descriptions['Reporting_date']),
                "predicate": "releaseDate"
            }
        )

        for aspect in out_ts.keys():
            processed_triples_dict.append(
                {
                    "subject": aspect,
                    "object": out_ts[aspect],
                    "predicate": "hasDescription"
                }
            )

        to_json("../results/processed/"+report_file+".json", processed_triples_dict)
        print("Finished")
