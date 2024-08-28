# Structuring Sustainability Reports for Environmental Standards with LLMs guided by Ontology
*by Aida Usmanova and Ricardo Usbeck*

This paper is part of the proceedings of the first ClimateNLP workshop at ACL 2024. The paper is available [here](https://aclanthology.org/2024.climatenlp-1.13/).

### Abstract
Following the introduction of the European Sustainability Reporting Standard (ESRS), companies will have to adapt to a new policy and provide mandatory sustainability reports. However, implementing such reports entails a challenge, such as the comprehension of a large number of textual information from various sources. This task can be accelerated by employing Large Language Models (LLMs) and ontologies to effectively model the domain knowledge. In this study, we extended an existing ontology to model ESRS Topical Standard for disclosure. The developed ontology would enable automated reasoning over the data and assist in constructing Knowledge Graphs (KGs). Moreover, the proposed ontology extension would also help to identify gaps in companies’ sustainability reports with regard to the ESRS requirements.Additionally, we extracted knowledge from corporate sustainability reports via LLMs guided with a proposed ontology and developed their KG representation.

### Usage:
1. Create a virtual environment `python -m venv venv` and install all requirements `pip install -r requirements.txt`
2. The `data/` folder contains sustainability reports. In this study we used pre-processed reports from [Bronzini et.al 2024](https://github.com/saturnMars/derivingStructuredInsightsFromSustainabilityReportsViaLargeLanguageModels)
3. Create `results/` folder containing `processed/` and `raw/` subfolders
4. Extract triples with `src/extract_triples.py` and generate a knowledge graph from the report with `src/generate_kg.py`. Before running set up your OpenAI key in both files.
5. Once KG is generated, you can visualize it with `src/visualize_kg.py`
   
