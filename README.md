# Structuring Sustainability Reports for Environmental Standards with LLMs guided by Ontology
*by Aida Usmanova and Ricardo Usbeck*

This paper is part of the proceedings of the first ClimateNLP workshop at ACL 2024. The paper is available [here](https://aclanthology.org/2024.climatenlp-1.13/).

### Abstract
Following the introduction of the European Sustainability Reporting Standard (ESRS), companies will have to adapt to a new policy and provide mandatory sustainability reports. However, implementing such reports entails a challenge, such as the comprehension of a large number of textual information from various sources. This task can be accelerated by employing Large Language Models (LLMs) and ontologies to effectively model the domain knowledge. In this study, we extended an existing ontology to model ESRS Topical Standard for disclosure. The developed ontology would enable automated reasoning over the data and assist in constructing Knowledge Graphs (KGs). Moreover, the proposed ontology extension would also help to identify gaps in companies’ sustainability reports with regard to the ESRS requirements.Additionally, we extracted knowledge from corporate sustainability reports via LLMs guided with a proposed ontology and developed their KG representation.

---------
### Usage:
1. Installation
   ```bash
   git clone https://github.com/aidausmanova/esrs_kg.git
   ```
2. Environment setup
   ```bash
   python -m venv venv
   pip install -r requirements.txt
   ```
3. The `data/` folder contains sustainability reports. In this study we used pre-processed reports from [Bronzini et.al 2024](https://github.com/saturnMars/derivingStructuredInsightsFromSustainabilityReportsViaLargeLanguageModels)
4. Create `results/` folder containing `processed/` and `raw/` subfolders
5. Running
   
   Extract triples with
   ```bash
   src/extract_triples.py
   ```

   Generate a knowledge graph from the report with 
   ```bash
   src/generate_kg.py
   ```

   **Note**: before running set up your OpenAI key in both files.
8. Once KG is generated, you can visualize it with
   ```bash
   src/visualize_kg.py
   ```
   
