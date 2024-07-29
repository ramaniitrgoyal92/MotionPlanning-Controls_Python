import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

show_animation = True
save_animation = True

class AStarPlanner:

    def __init__(self,ox,oy,resolution,turn_rad):
        self.ox = ox
        self.oy = oy
        self.min_x = round(min(ox))
        self.min_y = round(min(oy))
        self.max_x = round(max(ox))
        self.max_y = round(max(oy))
        
        self.resolution = resolution
        self.turn_rad = turn_rad
        self.width_x = round((self.max_x - self.min_x) / self.resolution)
        self.width_y = round((self.max_y - self.min_y) / self.resolution)
        self.motion = self.get_motion()
        self.gen_obstacle_map()

    class Node:
        def __init__(self,id_x,id_y,cost,parentIndex):
            self.id_x = id_x
            self.id_y = id_y
            self.cost = cost
            self.parentIndex = parentIndex
        
        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(
                self.cost) + "," + str(self.parent_index)

    def get_xy_index(self,pos,min_pos):
        return round((pos-min_pos)/self.resolution)
    
    def calc_grid_position(self, index, min_position):
        return (index * self.resolution + min_position)

    def planning(self,sx,sy,gx,gy):

        startNode = self.Node(self.get_xy_index(sx,self.min_x),
                              self.get_xy_index(sy,self.min_y),0.0,-1)
        
        goalNode = self.Node(self.get_xy_index(gx,self.min_x),
                              self.get_xy_index(gy,self.min_y),0.0,-1)

        openset = dict()
        closedset = dict()
        openset[self.get_node_idx(startNode)] = startNode

        while len(openset)!=0:
            current_idx = min(openset, key=lambda o: openset[o].cost+self.heuristic(openset[o],goalNode))
            currentNode = openset[current_idx]

            if show_animation:
                plt.plot(self.calc_grid_position(currentNode.id_x, self.min_x),
                         self.calc_grid_position(currentNode.id_y, self.min_y),"xc")
                if len(closedset.keys()) % 100 == 0:
                    plt.pause(0.0001)

            
            if currentNode.id_x == goalNode.id_x and currentNode.id_y == goalNode.id_x:
                print("Found Goal!")
                # goalNode = currentNode
                break

            del openset[current_idx]

            closedset[current_idx] = currentNode

            for i, _ in enumerate(self.motion):
                newNode = self.Node(currentNode.id_x+self.motion[i][0],
                                currentNode.id_y+self.motion[i][1],
                                currentNode.cost+self.motion[i][2],
                                self.get_node_idx(currentNode))
                
                if self.outside_or_hit_obs(newNode):
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

    def outside_or_hit_obs(self,newNode):       
        px = self.calc_grid_position(newNode.id_x, self.min_x)
        py = self.calc_grid_position(newNode.id_y, self.min_y)

        if px < self.min_x:
            return True
        elif py < self.min_y:
            return True
        elif px >= self.max_x:
            return True
        elif py >= self.max_y:
            return True

        if self.obstacle_map[newNode.id_x][newNode.id_y]:
            return True
        
        return False


    def get_final_path(self,goalNode,closedset):
        rx = [self.calc_grid_position(goalNode.id_x,self.min_x)]
        ry = [self.calc_grid_position(goalNode.id_y,self.min_y)]
        parentIndex = goalNode.parentIndex
        while parentIndex!=-1:
            rx.append(self.calc_grid_position(closedset[parentIndex].id_x,self.min_x))
            ry.append(self.calc_grid_position(closedset[parentIndex].id_y,self.min_y))
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
        return ((node.id_y- self.min_y)*self.width_x+(node.id_x- self.min_x))

    def heuristic(self,node1,node2):
        return math.sqrt((node1.id_x-node2.id_x)**2+(node1.id_y-node2.id_y)**2)

    def gen_obstacle_map(self):
    # obstacle map generation
        self.obstacle_map = [[False for _ in range(self.width_y)]
                             for _ in range(self.width_x)]
        for ix in range(self.width_x):
            x = self.calc_grid_position(ix, self.min_x)
            for iy in range(self.width_y):
                y = self.calc_grid_position(iy, self.min_y)
                for iox, ioy in zip(self.ox, self.oy):
                    d = math.hypot(iox - x, ioy - y)
                    if d <= self.turn_rad:
                        self.obstacle_map[ix][iy] = True
                        break


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

    planner = AStarPlanner(ox,oy,resolution=2.0,turn_rad=1.0)
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
                ax.plot(planner.calc_grid_position(currentNode.id_x, planner.min_x),
                         planner.calc_grid_position(currentNode.id_y, planner.min_y), "xc")
                plt.grid(True)
                plt.axis("equal")
            if frame == len(closedset)-1:
                plt.plot(rx, ry, "-r")

        ani = animation.FuncAnimation(fig, update, frames=len(closedset), interval = 1, repeat=False)        
        ani.save('animation_astar.gif', writer=PillowWriter(fps=50))
        plt.show()


if __name__ == '__main__':
    main()