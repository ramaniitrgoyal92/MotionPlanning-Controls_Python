import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

show_animation = True
save_animation = False

class AStarPlanner:

    def __init__(self,ox,oy):
        self.ox = ox
        self.oy = oy
        self.min_x = round(min(ox))
        self.min_y = round(min(oy))
        self.max_x = round(max(ox))
        self.max_y = round(max(oy))

        self.width_x = round((self.max_x - self.min_x))
        self.width_y = round((self.max_y - self.min_y))
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

        openset = dict()
        closedset = dict()
        openset[self.get_node_idx(startNode)] = startNode

        while len(openset)!=0:
            current_idx = min(openset, key=lambda o: openset[o].cost+self.heuristic(openset[o],gx,gy))
            currentNode = openset[current_idx]

            if show_animation:
                plt.plot(currentNode.x,currentNode.y,"xc")
                if len(closedset.keys()) % 100 == 0:
                    plt.pause(0.0001)

            
            if currentNode.x == gx and currentNode.y == gy:
                print("Found Goal!")
                # goalNode = currentNode
                break

            del openset[current_idx]

            closedset[current_idx] = currentNode

            for i, _ in enumerate(self.motion):
                newNode = self.Node(currentNode.x+self.motion[i][0],
                                currentNode.y+self.motion[i][1],
                                currentNode.cost+self.motion[i][2],
                                self.get_node_idx(currentNode))
                
                if self.hit_obst(newNode):
                    continue

                newNode_idx = self.get_node_idx(newNode)
                if newNode_idx in closedset:
                    continue
                    
                if newNode not in openset:    
                    openset[newNode_idx] = newNode
                else:
                    if openset[newNode_idx].cost > newNode.cost:
                        openset[newNode_idx] = newNode

        rx, ry = self.get_final_path(currentNode,closedset)
        return rx, ry, closedset

    def hit_obst(self,newNode):
        if (newNode.x, newNode.y) in set(zip(self.ox, self.oy)):
            return True
        else:
            return False
        


    def get_final_path(self,goalNode,closedset):
        rx, ry = [goalNode.x], [goalNode.y]
        parentIndex = goalNode.parentIndex
        while parentIndex!=-1:
            rx.append(closedset[parentIndex].x)
            ry.append(closedset[parentIndex].y)
            parentIndex = closedset[parentIndex].parentIndex
            
        return rx, ry    

    @staticmethod
    def get_motion():
        motion = [[0,1,1],[1,0,1],[-1,0,1],[0,-1,1],
                  [-1, -1, math.sqrt(2)],
                  [-1, 1, math.sqrt(2)],
                  [1, -1, math.sqrt(2)],
                  [1, 1, math.sqrt(2)]]
        return motion

    def get_node_idx(self,node):
        return ((node.y- self.min_y)*self.width_x+(node.x- self.min_x))

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

    planner = AStarPlanner(ox,oy)
    rx, ry, closedset = planner.planning(sx,sy,gx,gy)

    print(len(closedset))

    if show_animation:  # pragma: no cover
        plt.plot(rx, ry, "-r")
        plt.pause(0.001)
        plt.show()

    if save_animation:
        fig, ax = plt.subplots()
        ax.plot(ox, oy, ".k")
        ax.plot(sx, sy, "og")
        ax.plot(gx, gy, "xb")
        plt.grid(True)
        plt.axis("equal")

        def update(frame):
            if frame % 1 == 0:
                key = list(closedset.keys())[frame]
                currentNode = closedset[key]
                ax.plot(currentNode.x, currentNode.y, "xc")
                plt.grid(True)
                plt.axis("equal")
            if frame == len(closedset)-1:
                plt.plot(rx, ry, "-r")

        ani = animation.FuncAnimation(fig, update, frames=len(closedset), interval = 1, repeat=False)        
        ani.save('animation_astar.gif', writer=PillowWriter(fps=50))
        plt.show()


if __name__ == '__main__':
    main()