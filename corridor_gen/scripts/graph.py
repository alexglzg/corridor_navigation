class Graph:
    """
    A simple graph class.
    """

    def __init__(self, directed: bool = False) -> None:
        """
        Initializes a new graph.

        Parameters:
        - directed (bool): If True, the graph is directed; otherwise, it is undirected.
        """
        self._adj = {}
        self.directed = directed

    def add_node(self, node: any) -> None:
        """
        Adds a node to the graph.

        Parameters:
        - node: The node to be added. It can be any hashable type.
        """
        # Ignore if node already exists
        if node in self._adj:
            return
        if node not in self._adj:
            self._adj[node] = set()

    def add_edge(self, u: any, v: any) -> None:
        """
        Adds an edge between nodes u and v.

        Parameters:
        - u: The first node.
        - v: The second node.
        """
        if u not in self._adj:
            self._adj[u] = set()
        if v not in self._adj:
            self._adj[v] = set()
        self._adj[u].add(v)
        if not self.directed:
            self._adj[v].add(u)

    def remove_edge(self, u: any, v: any) -> None:
        """
        Removes the edge between nodes u and v.

        Parameters:
        - u: The first node.
        - v: The second node.
        """
        if u in self._adj and v in self._adj[u]:
            self._adj[u].remove(v)
        if not self.directed and v in self._adj and u in self._adj[v]:
            self._adj[v].remove(u)

    def remove_node(self, node: any) -> None:
        """
        Removes a node and all its edges from the graph.

        Parameters:
        - node: The node to be removed.
        """
        if node in self._adj:
            del self._adj[node]
        for neighbors in self._adj.values():
            neighbors.discard(node)

    def neighbors(self, node: any) -> set:
        """
        Returns a copy of the set of neighbors for a given node.

        Parameters:
        - node: The node whose neighbors are to be returned.
        Returns:
        - A set of neighbors for the specified node. If the node does not exist, returns an empty set.
        """
        return self._adj.get(node, set()).copy()

    def nodes(self) -> set:
        """
        Returns a set of all nodes in the graph.

        Returns:
        - A set containing all nodes in the graph.
        """
        return set(self._adj.keys())

    def edges(self) -> set:
        """
        Returns a set of all edges in the graph.

        Returns:
        - A set of tuples representing edges in the graph. Each edge is represented as a tuple (u, v).
        """
        edges = set()
        for u, neighbors in self._adj.items():
            for v in neighbors:
                if self.directed or (v, u) not in edges:
                    edges.add((u, v))
        return edges

    def __len__(self) -> int:
        """
        Returns the number of nodes in the graph.

        Returns:
        - The number of nodes in the graph.
        """
        return len(self._adj)
