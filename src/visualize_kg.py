from pyvis.network import Network
import networkx as nx
import gradio as gr
import json


def read_json(path):
    with open(path, 'r') as f:
        content = json.load(f)
    return content

def create_graph_from_triplets(triplets_dict):
    G = nx.DiGraph()
    # for triplet in triplets:
    #     subject, predicate, obj = triplet.strip().split('~')
    #     G.add_edge(subject.strip(), obj.strip(), label=predicate.strip())
    # for triplet in triplets:
    #     subject, predicate, obj = triplet.strip().split('|')
    for item in triplets_dict:
        G.add_edge(item['subject'], item['object'], label=item['predicate'])
        
    return G

def nx_to_pyvis(networkx_graph):
    pyvis_graph = Network(notebook=True, cdn_resources='remote')
    for node in networkx_graph.nodes():
        pyvis_graph.add_node(node)
    for edge in networkx_graph.edges(data=True):
        pyvis_graph.add_edge(edge[0], edge[1], label=edge[2]["label"])
    return pyvis_graph

def generateGraph(json_file):
    triples_dict = read_json("../results/processed/"+json_file+".json")
    # triplets = [t.strip() for t in triples_list_clean if t.strip()]
    graph = create_graph_from_triplets(triples_dict)
    pyvis_network = nx_to_pyvis(graph)

    pyvis_network.toggle_hide_edges_on_drag(True)
    pyvis_network.toggle_physics(False)
    pyvis_network.set_edge_smooth('discrete')

    html = pyvis_network.generate_html()
    html = html.replace("'", "\"")

    return f"""<iframe style="width: 100%; height: 600px;margin:0 auto" name="result" allow="midi; geolocation; microphone; camera;
    display-capture; encrypted-media;" sandbox="allow-modals allow-forms
    allow-scripts allow-same-origin allow-popups
    allow-top-navigation-by-user-activation allow-downloads" allowfullscreen=""
    allowpaymentrequest="" frameborder="0" srcdoc='{html}'></iframe>"""


demo = gr.Interface(
        generateGraph,
        inputs="text",
        outputs=gr.outputs.HTML(),
        title="Knowledge Graph",
        allow_flagging='never',
        live=True,
    )



if __name__ == "__main__":
    # Go to http://127.0.0.1:7860/

    demo.launch(
        height=800,
        width="100%"
    )