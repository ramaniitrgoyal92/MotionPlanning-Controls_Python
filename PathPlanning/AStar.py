import numpy as np
import matplotlib.pyplot as plt
import math

show_animation = True


class AStarPlanner:

    def __init__(self,resolution,width_x):
        self.resolution = resolution
        self.width_x = width_x
        self.motion = self.get_motion()

    class Node:
        def __init__(self,x,y,cost,parentIndex):
            self.x = x
            self.y = y
            self.cost = cost
            self.parentIndex = parentIndex
        
        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(
                self.cost) + "," + str(self.parent_index)

    def planning(self,sx,sy,gx,gy):

        startNode = self.Node(sx,sy,0.0,-1)

        visited = set()
        visited.add(startNode)

        openset = dict()
        openset[self.get_node_idx(startNode)] = startNode

        while len(openset)!=0:
            current_idx = min(openset, key=lambda o: openset[o].cost+self.heuristic(openset[o],gx,gy))
            currentNode = openset[current_idx]

            if currentNode.x != gx and currentNode.y != gy:
                break

            visited.add(currentNode)

            for i, _ in enumerate(self.motion):
                newNode = self.node(currentNode.x+self.motion[i][0],
                                currentNode.y+self.motion[i][1],
                                currentNode.cost+self.motion[i][2],
                                self.get_node_idx(currentNode))
                # newNode.cost = currentNode.cost + cost_matrix[self.get_node_idx(currentNode)][self.get_node_idx(newNode)]
                # newNode.heuristic = newNode.cost + self.heuristic(newNode)
                # newNode.parentNode = currentNode
                    
                if newNode not in visited:    
                    openset[self.get_node_idx(newNode)] = newNode

    @staticmethod
    def get_motion():
        motion = [[0,1,1],[1,0,1],[-1,0,1],[0,-1,1],
                  [-1, -1, math.sqrt(2)],
                  [-1, 1, math.sqrt(2)],
                  [1, -1, math.sqrt(2)],
                  [1, 1, math.sqrt(2)]]
        return motion

    def get_node_idx(self,node):
        return (node.y*self.width_x+node.x)

    def heuristic(self,node,gx,gy):
        return math.sqrt((node.x-gx)**2+(node.y-gy)**2)




def main():
    print("Start!")
    # start and goal position
    sx = 10.0  # [m]
    sy = 10.0  # [m]
    gx = 50.0  # [m]
    gy = 50.0  # [m]

    # set obstacle positions
    ox, oy = [], []
    for i in range(-10, 60):
        ox.append(i)
        oy.append(-10.0)
    for i in range(-10, 60):
        ox.append(60.0)
        oy.append(i)
    for i in range(-10, 61):
        ox.append(i)
        oy.append(60.0)
    for i in range(-10, 61):
        ox.append(-10.0)
        oy.append(i)
    for i in range(-10, 40):
        ox.append(20.0)
        oy.append(i)
    for i in range(0, 40):
        ox.append(40.0)
        oy.append(60.0 - i)

    if show_animation:  # pragma: no cover
        plt.plot(ox, oy, ".k")
        plt.plot(sx, sy, "og")
        plt.plot(gx, gy, "xb")
        plt.grid(True)
        plt.axis("equal")

    planner = AStarPlanner(resolution=10,width_x = 10)
    rx, ry = planner.planning(sx,sy,gx,gy)

    if show_animation:  # pragma: no cover
        plt.plot(rx, ry, "-r")
        plt.pause(0.001)
        plt.show()


if __name__ == '__main__':
    main()

