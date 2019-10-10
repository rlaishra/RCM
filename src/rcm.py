from __future__ import division
import networkx as nx
import numpy as np
from numpy import linalg as LA
from multiprocessing import Pool
import multiprocessing
from scipy.sparse import csr_matrix


class RCM(object):
	"""docstring for RCM"""
	def __init__(self, graph, theta, budget, kcore=None):
		super(RCM, self).__init__()
		self._graph = graph
		if kcore is not None:
			self._kcore = kcore
		else:
			self._kcore = nx.core_number(self._graph)

		self._theta = theta
		self._budget = budget

		self._residualDegree()
		self._candidateNodes()

		self._graph = self._graph.subgraph(self._ca.union(self._cf)).copy()

		self._sg = self._graph.subgraph(self._cf)
		self._cc = [self._sg.subgraph(c).copy() for c in nx.connected_components(self._sg)]


	def _residualDegree(self):
		self._delta = {}
		for u in self._kcore:
			d = self._theta - len([v for v in self._graph[u] if self._kcore[v] >= self._theta])
			if d > 0:
				self._delta[u] = d


	def _candidateNodes(self):
		self._cf = set([u for u in self._kcore if self._kcore[u] < self._theta and len(self._graph[u]) >= self._theta])
		self._ca = set([u for u in self._kcore if self._kcore[u] < self._theta and len(self._cf.intersection(self._graph[u])) > 0])


	def _residualAnchors(self, c, c1, a0, tdelta):
		a = set([])
		while len(c1) > 0:
			n = {u:len(c1.intersection(self._graph[u])) for u in a0}
			mval = max(n.values())
			n = [u for u in n if n[u] == mval]
			if len(n) > 1:
				n = {u: len(self._cf.intersection(self._graph[u])) for u in n}
				mval = max(n.values())
				n = [u for u in n if n[u] == mval]
				np.random.shuffle(n)
			v = n[0]
			tdelta.update({u: tdelta[u] - 1 for u in c1.intersection(self._graph[v])})
			r = [u for u in c1 if tdelta[u] <= 0]
			c1.difference_update(r)
			a.update([v])
			a0.remove(v)
		return [{'anchors':a, 'followers':set(list(c.nodes()))}]


	def _anchorScoreAnchors(self, c):
		mfol, manch, mc = set([]), set([]), 0

		cf0 = set(c.nodes())
		neighbors = set([v for u in cf0 for v in self._graph[u]])
		ca0 = neighbors.intersection(self._ca)
		tdelta = {u: self._delta[u] for u in ca0.union(cf0)}

		g = self._graph.subgraph(cf0.union(ca0))
		tfols = set([])
		tanchors = set([])

		solutions = []

		while len(tanchors) < self._budget and len(cf0) > 0 and len(ca0) > 0:
			candidates = self._anchorScore(g, ca0, cf0, tdelta)
			if candidates is None or len(candidates) == 0:
				break

			if len(candidates) > 1:
				dc = {u:len(cf0.intersection(g[u])) for u in candidates}
				mval = max(dc.values())
				candidates = [u for u in dc if dc[u] == mval]
				u = np.random.choice(candidates)
			else:
				u = candidates[0]

			g, fol, ca0, cf0, tdelta = self._updateCore(u, tanchors, ca0, cf0, tdelta)
			tanchors.update([u])
			tfols.update(fol)

			solutions.append({'anchors':set(tanchors), 'followers':set(tfols)})
				
		return solutions


	def _anchorScoreSR(self, g, nodes0, nodes1):
		score, tnodes0, tnodes, tinodes = {}, {}, {}, {}
		nodes = list(g.nodes())
		inodes = {nodes[k]:k for k in range(len(nodes))}
		
		M, V = [], []

		row = [0 for _ in nodes]
		for u in nodes:
			n = set(g[u])
			fol = set([v for v in g[u] if v in self._delta and self._delta[v] == 1])
			trow = list(row)
			for v in fol.intersection(nodes):
				trow[inodes[v]] = 1.0
			for v in n.difference(fol):
				trow[inodes[v]] = np.round(1/self._delta[v],1)			# approximation

			trow.append(len(fol.difference(nodes)) + 1)
			M.append(trow)

		M.append([0 for _ in range(len(nodes))])
		M[-1].append(1)

		M = np.array(M, dtype=np.float32)
		M = csr_matrix(M)
		for _ in range(0, 3):
			M = M.dot(M)

		for i in range(len(nodes)):
			score[nodes[i]] = np.sum(M[i,])

		return score


	def _anchorScore(self, g, ca, cf, tdelta):
		mnodes, mval = [], -1

		score = {u:1 for u in ca.union(cf)}					# Assign a default score to every node
		nodes0 = cf.difference(ca)
		nodes1 = ca.intersection(cf)

		score.update(self._anchorScoreSR(g, nodes0, nodes1))
		
		nodes2 = ca.difference(cf)
		nodse2 = nodes2.intersection(g.nodes())
		
		for u in nodes2:
			if u not in g.nodes():
				continue
			fol = set([v for v in g[u] if v in self._delta and self._delta[v] == 1])
			s = 1 + np.sum([score[v] for v in fol])
			
			s += np.sum([score[v]/self._delta[v] for v in cf.intersection(g[u]).difference(fol)])

			if s >= mval:
				mval = s
				mnodes = [u]
			elif s == mval:
				mnodes.append(u)

		return mnodes


	def _findResidualCore(self, sg, delta, followers=None):
		if followers is not None:
			print(followers)
			queue = set([v for u in followers for v in self._cf.intersection(self._graph[u])])
			seen = set(queue)

			while len(queue) > 0:
				u = queue.pop()
				n = self._cf.intersection(set(self._graph[u]).difference(seen))
				seen.update(n)
				queue.update(n)
			seen.difference_update(followers)
			fsg = nx.Graph(self._sg.subgraph(seen))
		else:
			fsg = nx.Graph(self._sg.subgraph(self._cf))

		while True:
			rnodes = [v for v in fsg.nodes() if len(fsg[v]) < self._delta[v]]
			fsg.remove_nodes_from(rnodes)
			if len(rnodes) == 0 or fsg.number_of_nodes() == 0:
				break

		fol = list(fsg.nodes())
		return fol


	def _updateCore(self, u, anchors, ca, cf, delta):
		queue = set([u])
		followers = set([u])
		tdelta = dict(delta)
		tca = set(ca)
		tcf = set(cf)
		tsg = self._sg.copy()

		del tdelta[u]

		tca.difference_update(followers)
		tcf.difference_update(followers)

		first = True
		while len(queue) > 0:
			f = set([])
			if first:
				f = set([u])
				first = False
			while len(queue) > 0:
				current = queue.pop()
				#print(current)
				#print(self._sg[current])
				n = cf.intersection(self._graph[current])
				
				for v in n:
					if v not in tdelta:
						continue
					tdelta[v] -= 1
					
					if tdelta[v] == 0:
						queue.update([v])
						f.update([v])
				
			tcf.difference_update(f)
			tca.difference_update(f)
			followers.update(f)

			sc = self._findResidualCore(tsg, tdelta, f)

			followers.update(sc)
			queue.update(sc)
			
			tcf.difference_update(sc)
			tca.difference_update(sc)

			for v in followers:
				try:
					del tdelta[v]
				except:
					pass

		followers = set(followers)
		tsg.remove_nodes_from(anchors)
		tsg.remove_nodes_from(followers)

		followers.difference_update([u])
		followers.difference_update(anchors)

		return tsg, followers, tca, tcf, tdelta


	def _rcmThread(self, arg):
		c = self._cc[arg]

		c0 = set([u for u in c.nodes() if len(c[u]) >= self._delta[u]])
		c1 = set(c.nodes()).difference(c0)
		a0 = set([u for u in self._ca.difference(c0.union(c1)) if len(c1.intersection(self._graph[u])) > 0])
		tdelta = {u:self._delta[u] - len(c[u]) for u in c1}

		lower_bound = max(tdelta.values())
		upper_bound = sum(tdelta.values())
		threshold = min(tdelta.values())

		if threshold > self._budget:
			return (None, None)

		if upper_bound < self._budget:
			solutions = self._residualAnchors(c, c1, a0, tdelta)
			return (solutions, None, None)
		elif lower_bound > self._budget :
			solutions = self._anchorScoreAnchors(c)
			return (None, solutions, None)
		else:
			s0, s1 = {}, {}
			s0 = self._residualAnchors(c, c1, a0, tdelta)
			if len(s0[0]['followers']) == len(c.nodes()) and len(s0[0]['anchors']) <= self._budget:
				return (s0, None)
			
			s1 = self._anchorScoreAnchors(c)
			if len(s0[0]['followers']) < len(s1[-1]['followers']) or len(s0[0]['anchors']) > self._budget:
				return (None, s1)
			else:
				return (s0, None)


	def _solutionSelectionBest(self, a, f, solutions):
		msol, mval = None, -1
		rfol = set([])
		for s in solutions:
			n = len(s['followers'].difference(f))
			m = len(s['anchors'].difference(a))

			if m == 0 and n > 0:
				rfol.update(s['followers'].difference(f))

			if n > 0 and m > 0 and n/m > mval and len(a.union(s['anchors'])) <= self._budget:
				mval = n/m
				msol = s
		return msol, rfol


	def _solutionSelection(self, approx_solutions, exact_solutions):
		solutions = exact_solutions
		solutions += approx_solutions

		a, f = set([]), set([])
		
		while len(a) < self._budget:
			s, r = self._solutionSelectionBest(a, f, solutions)
			if s is None:
				break
			a.update(s['anchors'])
			f.update(s['followers'])
			f.update(r)

		return a, f


	def _rcmParallel(self):
		with Pool() as pool:
			results = pool.map(self._rcmThread, range(len(self._cc)))

		approx_solutions = []			# these can be split
		exact_solutions = []			# these cannot be split

		for r in results:
			if r[0] is not None:
				exact_solutions += r[0]
			if r[1] is not None:
				approx_solutions += r[1]
		
		anchors, followers = self._solutionSelection(approx_solutions, exact_solutions)

		return anchors, followers


	def _rcmSequential(self):
		approx_solutions, exact_solutions = [], []

		for i in range(len(self._cc)):
			s0, s1 = self._rcmThread(i)

			if s0 is not None:
				exact_solutions += s0
			if s1 is not None:
				approx_solutions += s1

		anchors, followers = self._solutionSelection(approx_solutions, exact_solutions)

		return anchors, followers



	def findAnchors(self, parallel=True):
		if parallel:
			return self._rcmParallel()
		else:
			return self._rcmSequential()


if __name__ == '__main__':
	fname = sys.argv[1]
	theta = int(sys.argv[2])
	budget = float(sys.argv[3])

	rcm(fname, theta, budget)
