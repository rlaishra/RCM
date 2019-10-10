import networkx as nx



class Olak(object):
	def __init__(self, anchors=[]):
		super(Olak, self).__init__()
		self._anchors = anchors

	def onionPeeling(self, graph, k, anchors=[], core_numbers=None):
		i = 0
		
		if core_numbers is None:
			N = nx.k_core(graph, k=k-1)
		else:
			nodes = [u for u in core_numbers if core_numbers[u] >= k-1 or u in anchors]
			N = nx.Graph(graph.subgraph(nodes))

		P = [u for u in N.nodes() if len(N[u]) < k and u not in anchors]
		L = {}
		L_nodes = {}
		
		while len(P) > 0:
			i += 1
			L[i] = [v for v in P if v not in anchors]
			L_nodes.update({x:i for x in P})
			N.remove_nodes_from({u:i for u in P})
			P = [u for u in N.nodes() if len(N[u]) < k]
		
		N_nodes = set(list(N.nodes()))

		L[0] = []
		for u in L_nodes:
			L[0] += [v for v in graph[u] if v not in L_nodes and v not in N_nodes and v not in anchors]

		L_nodes.update({u:0 for u in L[0]})
		
		return L, L_nodes


	def upperBoundPrunnig(self, graph, L, L_nodes):
		W = {}
		UB = {u:0 for u in L_nodes}
		lower = []

		for u in L_nodes:
			W[u] = [v for v in graph[u] if v in L_nodes and L_nodes[v] > L_nodes[u]]

		N = sorted(L_nodes, key=L_nodes.get, reverse=True)
		for u in N:
			if len(W[u]) == 0:
				UB[u] = 0
			else:
				UB[u] = sum([(UB[v] +1) for v in W[u]])
		return UB


	def  supportExist(self, graph, n, x, L_nodes):
		l = L_nodes[x]
		n0 = set(list(graph[x])).intersection(L_nodes.keys())
		n0 = n0.intersection(n)
		for y in n0:
			if L_nodes[y] > L_nodes[x] or L_nodes[y] == L_nodes[x]:
				return True
		return False


	def findFollowers(self, graph, L, L_nodes, u):
		F = [u]
		Q = set([u])
		S = set([u])
		l = L_nodes[u]
		
		while len(Q) > 0:
			v = Q.pop()
			n = []

			s = {x:L_nodes[x] for x in graph[v] if x in L_nodes and L_nodes[x] > 0}
			s = sorted(s, key=s.get, reverse=False)
		
			for x in s:
				if x not in L_nodes:
					continue

				if self.supportExist(graph, F, x, L_nodes):
					n.append(x)
					if x not in Q and x not in S:
						Q.update([x])
						S.update([x])

			F += n
		return set(F)
		

	# OLAK
	def olak(self, graph, k_anchor, anchors=[], olak_kcore=[]):
		L, L_nodes = self.onionPeeling(graph, k_anchor, anchors, olak_kcore)
		U = self.upperBoundPrunnig(graph, L, L_nodes)
		F = []
		Lambda = 0

		anchor = []

		T = sorted(U, key=U.get, reverse=True)

		while len(T) > 0:
			u = T.pop(0)

			if u in anchors or olak_kcore[u] >= k_anchor:
				continue
			
			if U[u] < Lambda :
				break
			
			F = self.findFollowers(graph, L, L_nodes, u)
			F = F.difference(anchors)
			
			if len(F) == 0:
				pass

			if len(F) > Lambda:
				Lambda = len(F)
				anchor = [u]

			break
		
		if len(anchor) > 0:
			anchors.update([anchor[0]])
			return anchors

		return anchors
		

# ## Anchored k-Core Decomposition
def anchoredKCore(graph, anchors=[], stop=-1, start=0, aval=-1):
	kcore = {u:-1 for u in graph.nodes()} 									# anchors gets a core number of -1
	sg = nx.Graph(graph)

	k = start
	cnodes = set(graph.nodes()) 											# nodes left in the core
	cnodes.difference_update(anchors)
	
	while True:
		# Remove unanchored nodes with degree less than k
		rnodes = [u for u in cnodes if len(sg[u]) < k]
		cnodes.difference_update(rnodes)
		sg.remove_nodes_from(rnodes)
		
		if len(rnodes) == 0:
			for u in cnodes:
				kcore[u] = k
			k += 1

		if (stop > 0 and k > stop):
			for u in cnodes:
				kcore[u] = k
			break

		if len(cnodes) == 0:
			break

	for u in anchors:
		kcore[u] = aval

	return kcore


def getAnchorsFollowersSG(graph, kcore, theta):
	tanchors = set([u for u in kcore if kcore[u] >= theta])
	cf = set([u for u in kcore if kcore[u] < theta and len(graph[u]) >= theta])				
	ca = set([u for u in kcore if kcore[u] < theta and len(cf.intersection(graph[u])) > 0])

	sg = nx.Graph(graph.subgraph(cf.union(ca).union(tanchors)))

	return sg


def olakAnchors(graph, kcore, theta, budget):
	sg = getAnchorsFollowersSG(graph, kcore, theta)

	olak_anchors = set(tanchors)
	olak_kcore = anchoredKCore(sg, anchors=olak_anchors, start=theta-1, stop=theta, aval=theta)
	
	for _ in range(budget):
		ol = Olak(olak_anchors)
		olak_anchors = ol.olak(sg, theta, olak_anchors, olak_kcore)
		olak_kcore = anchoredKCore(sg, anchors=olak_anchors, start=theta-1, stop=theta, aval=theta)
	
	return olak_anchors



