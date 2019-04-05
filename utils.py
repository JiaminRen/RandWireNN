import networkx as nx
import collections


Node = collections.namedtuple('Node', ['id', 'inputs', 'type'])

def get_graph_info(graph):
  input_nodes = []
  output_nodes = []
  Nodes = []
  for node in range(graph.number_of_nodes()):
    tmp = list(graph.neighbors(node))
    tmp.sort()
    type = -1
    if node < tmp[0]:
      input_nodes.append(node)
      type = 0
    if node > tmp[-1]:
      output_nodes.append(node)
      type = 1
    Nodes.append(Node(node, [n for n in tmp if n < node], type))
  return Nodes, input_nodes, output_nodes

def build_graph(Nodes, args):
  if args.graph_model == 'ER':
    return nx.random_graphs.erdos_renyi_graph(Nodes, args.P, args.seed)
  elif args.graph_model == 'BA':
    return nx.random_graphs.barabasi_albert_graph(Nodes, args.M, args.seed)
  elif args.graph_model == 'WS':
    return nx.random_graphs.connected_watts_strogatz_graph(Nodes, args.K, args.P, tries=200, seed=args.seed)

def save_graph(graph, path):
  nx.write_yaml(graph, path)

def load_graph(path):
  return nx.read_yaml(path)