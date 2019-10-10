import networkx as nx
import rcm
import sys

def readGraph(fname):
	if fname.endswith('mtx'):
		edges = []
		with open(fname, 'r') as f:
			reader = csv.reader(f, delimiter=' ')
			edges = [row for row in reader if len(row) == 2]
			f.close()
		graph = nx.Graph()
		graph.add_edges_from(edges, nodetype=int)
	else:
		graph = nx.read_edgelist(fname)
	
	graph.remove_edges_from(graph.selfloop_edges())
	graph.remove_nodes_from(list(nx.isolates(graph)))

	print(nx.info(graph))
	
	return graph


if __name__ == '__main__':
	fname = sys.argv[1]
	theta = int(sys.argv[2])
	budget = int(sys.argv[3])

	graph = readGraph(fname)

	r = rcm.RCM(graph, theta, budget)
	a, f = r.findAnchors()

	print(a,f)