#import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
from visualization import plotlyvisualize as plotly
from scipy.stats import entropy


def visualize_semantic_netwrok(topics, word_based_on_topic, visualize_method="networkx", filename="networkx", title=""):
    #graph = get_semantic_network2(topics, word_based_on_topic)
    graph = get_semantic_network(topics)
    weights = [graph[u][v]['weight'] for u, v in graph.edges()]
    node_size = nx.get_node_attributes(graph, 'importance').values()

    pos = nx.spring_layout(graph)

    for node, (x, y) in pos.items():
        graph.node[node]['x'] = float(x)
        graph.node[node]['y'] = float(y)

    nx.write_graphml(graph, "film_review_old_schema.graphml")

    if visualize_method == "networkx":
        #pos = nx.graphviz_layout(graph)
        try:
            pos = graphviz_layout(graph)
        except:
            pos = nx.random_layout(graph)

        nx.draw_networkx_nodes(graph, pos,
                           nodelist = graph.nodes(),
                           #node_color=node_color,
                           node_size=list(node_size),
                            alpha=1
                                )
        # edges
        weights = [weight/10 for weight in weights]
        nx.draw_networkx_edges(graph, pos, width=weights, alpha=0.5)
        nx.draw_networkx_labels(graph, pos, font_size=25, font_color='b')
        plt.show()

    elif visualize_method == "plotly":
        plotly.visualize(graph, node_size, weights, filename, title=title)
    else:
        raise visualize_method + " not defined for visualize method. use networkx or plotly as visualize_method"

def get_semantic_network(topics):
    graph = nx.Graph()
    for topic in topics:
        for (word1, prob1) in topic:
            for (word2, prob2) in topic:
                if word1 != word2:
                    graph.add_edge(word1, word2, weight = abs(prob1)*abs(prob2)*1000)
                    graph.add_node(word1, {"importance" : abs(prob1)*100})
                    graph.add_node(word2, {"importance" : abs(prob2)*100})

    return graph


def get_semantic_network2(topics, word_based_on_topic):
    graph = nx.Graph()
    for topic in topics:
        for (word1, prob1) in topic:
            for (word2, prob2) in topic:
                if word1 != word2:
                    #graph.add_edge(word1, word2, weight = abs(prob1)*abs(prob2)*1000)
                    kl_divergence = float(entropy(word_based_on_topic[word1], word_based_on_topic[word2]))
                    graph.add_edge(word1, word2, weight = kl_divergence)
                    graph.add_node(word1, {"importance" : float(abs(prob1)*100)})
                    graph.add_node(word2, {"importance" : float(abs(prob2)*100)})

    return graph