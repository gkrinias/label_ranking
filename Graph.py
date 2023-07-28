import numpy as np
from collections import defaultdict

 
class Graph():
  def __init__(self, vertices):
    self.graph = defaultdict(list)
    self.V = vertices

  def addEdge(self, u, v):
    self.graph[u].append(v)

  def isCyclicUtil(self, v, visited, recStack):
    # Mark current node as visited and
    # adds to recursion stack
    visited[v] = True
    recStack[v] = True

    # Recur for all neighbours
    # if any neighbour is visited and in
    # recStack then graph is cyclic
    for neighbour in self.graph[v]:
      if visited[neighbour] == False:
        if self.isCyclicUtil(neighbour, visited, recStack) == True:
          return True
      elif recStack[neighbour] == True:
        return True

    # The node needs to be popped from
    # recursion stack before function ends
    recStack[v] = False
    return False

  # Returns true if graph is cyclic else false
  def isCyclic(self):
    visited = [False] * self.V
    recStack = [False] * self.V
    for node in range(self.V):
      if visited[node] == False:
        if self.isCyclicUtil(node, visited, recStack) == True:
          return True
    return False
  
  def topologicalSortUtil(self, v, visited, stack):
    # Mark the current node as visited.
    visited[v] = True

    # Recur for all the vertices adjacent to this vertex
    for i in self.graph[v]:
      if visited[i] == False:
        self.topologicalSortUtil(i, visited, stack)

    # Push current vertex to stack which stores result
    stack.insert(0, v)
  
  def topologicalSort(self):
    # Mark all the vertices as not visited
    visited = [False] * self.V
    stack = []
 
    # Call the recursive helper function to store Topological
    # Sort starting from all vertices one by one
    for i in range(self.V):
      if visited[i] == False:
        self.topologicalSortUtil(i, visited, stack)
 
    # Print contents of stack
    return stack
  

def flatten(S):
  """
  Recursively flatten a list of lists of lists of...
  """
  if not S:
    return S
  if isinstance(S[0], list):
    return flatten(S[0]) + flatten(S[1:])
  return S[:1] + flatten(S[1:])


def kwikSort(V, A):
  """
  Approximation algorithm for constructing a permutation
  from tournament graph (pairwise orderings), so that the
  resultant conflicts are minimized.
  """
  if not V: return []
  Vl = set()
  Vr = set()
  i = np.random.choice(list(V))

  for j in V.difference({i}): Vl.add(j) if (j, i) in A else Vr.add(j)

  Al = set((i, j) for (i, j) in A if i in Vl and j in Vl)
  Ar = set((i, j) for (i, j) in A if i in Vr and j in Vr)

  return [kwikSort(Vl, Al), i, kwikSort(Vr, Ar)]


def tournament2ranking(n_vertices, arcs):
  # g = Graph(n_vertices)
  # for arc in arcs: g.addEdge(arc[0], arc[1])
  # if not g.isCyclic(): return g.topologicalSort()
  return flatten(kwikSort(set(range(n_vertices)), arcs))