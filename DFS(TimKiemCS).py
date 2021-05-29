import networkx as nx
import matplotlib.pyplot as plt
# Bai tap 2
def Depth_First_Search(initialState, goalTest):
    frontier = []
    frontier.append(initialState)
    explored = []
    while len(frontier) > 0:
        print("frontier: ", frontier)
        state = frontier.pop(len(frontier)-1)  # pop(0) [1,2,3,4,5]
        explored.append(state)
        if goalTest == state:
            return True
        for neighbor in G.neighbors(state):
            if neighbor not in list(set(frontier + explored)):
                frontier.append(neighbor)
    return False
if __name__ == "__main__":
    G = nx.Graph()
    G.add_nodes_from(["S", "A", "B", "C", "D", "E", "F", "G", "H"])
    G.add_weighted_edges_from(
        [
            ("S", "A", 1),
            ("S", "B", 1),
            ("S", "C", 1),
            ("A", "D", 1),
            ("B", "G", 1),
            ("B", "E", 1),
            ("C", "E", 1),
            ("D", "F", 1),
            ("E", "H", 1),
            ("F", "E", 1),
            ("H", "G", 1),
            ("F", "G", 1),
        ]
    )
    result = Depth_First_Search("S", "H")
    #print(result)