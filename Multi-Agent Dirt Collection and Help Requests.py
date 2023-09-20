#!/usr/bin/env python

# std library imports
import abc
import sys
import threading

# third party imports
import actionlib
import dynamic_reconfigure.client
import numpy as np
import matplotlib.pyplot as plt
import rospy
import tf
from nav_msgs.srv import GetMap
from move_base_msgs.msg import MoveBaseAction
from move_base_msgs.msg import MoveBaseGoal
from std_msgs.msg import String
from sklearn import svm

from shapely.geometry import Polygon, Point, LineString
import math
import heapq
from typing import Optional, List
import random
import matplotlib.patches as pat
import shapely.affinity as sa
from scipy.stats import norm

N = 100  # Environment Size is N*N
# Note: 0.0026 is the remainder form 3 standart diviations (%99.74)
# Note: @ is a symbol for multypling matricies

##################### Part 1 ###########################
# Helper Classes


class MapService(object):

    def __init__(self):
        """
        Class constructor
        """
        rospy.wait_for_service('static_map')
        static_map = rospy.ServiceProxy('static_map', GetMap)
        self.map_data = static_map().map
        self.map_org = np.array(
            [self.map_data.info.origin.position.x, self.map_data.info.origin.position.y])
        shape = self.map_data.info.height, self.map_data.info.width
        self.map_arr = np.array(
            self.map_data.data, dtype='float32').reshape(shape)
        self.resolution = self.map_data.info.resolution

        self.start = None

    def position_to_map(self, pos):
        return (pos - self.map_org) // self.resolution

    def map_to_position(self, indices):
        return indices * self.resolution + self.map_org


class Cleaner(object):
    def __init__(self, agent_id, agent_max_vel):
        self.agent_id = agent_id
        self.agent_max_vel = agent_max_vel
        self.adversary_id = 0 if agent_id == 1 else 1
        self.dirt_pieces = []
        self.map_service = MapService()
        self.curr_t = None
        self.curr_R = None
        self.adversary_t = None
        self.adversary_R = None

        self.location_listener = tf.TransformListener()

        rospy.Subscriber('dirt', String, self.update_dirt_locations)

        rc_DWA_client = dynamic_reconfigure.client.Client(
            "tb3_%d/move_base/DWAPlannerROS/" % self.agent_id)
        rc_DWA_client.update_configuration(
            {"max_vel_trans": self.agent_max_vel})
        rc_DWA_client.update_configuration({"max_vel_x": self.agent_max_vel})

        # Create an action client called "move_base" with action definition file "MoveBaseAction"
        self.client = actionlib.SimpleActionClient(
            'tb3_%d/move_base' % self.agent_id, MoveBaseAction)

        # Waits until the action server has started up and started listening for goals.
        self.client.wait_for_server()

        t = threading.Thread(target=self.run_cleaner)
        t.start()

    def update_dirt_locations(self, msg):
        raw = msg.data
        next_closure = raw.find(']')+1
        new_dirt_pieces = []
        while next_closure > 0:
            loc = (eval(raw[0:next_closure]))
            new_dirt_pieces.append(self.map_service.position_to_map(
                np.array([loc[1], loc[0]])).tolist())
            raw = raw[next_closure:]
            next_closure = raw.find(']')+1

        if new_dirt_pieces != self.dirt_pieces:
            self.dirt_pieces = new_dirt_pieces

    def run_cleaner(self):
        # wait to get all dirt pieces
        rate = rospy.Rate(20)
        while len(self.dirt_pieces) == 0:
            rospy.loginfo("Agent "+str(self.agent_id) +
                          " waiting for dirt pieces locations...")
            rate.sleep()

        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            try:
                # get your location and adversary location
                self.curr_t, self.curr_R = self.location_listener.lookupTransform(
                    '/map', '/tb3_%d/base_link' % self.agent_id, rospy.Time(0))
                self.adversary_t, self.adversary_R = self.location_listener.lookupTransform(
                    '/map', '/tb3_%d/base_link' % self.adversary_id, rospy.Time(0))

                # get next goal according to cleaner heuristic
                self.get_next_goal()

                rate.sleep()

            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue

    def go_to(self, x, y, w=1.0):
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()

        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        goal.target_pose.pose.orientation.w = 1.0

        # Sends the goal to the action server.
        self.client.send_goal(goal)
        rospy.loginfo("New goal command received!")

    @abc.abstractmethod
    def get_next_goal(self):
        """ cleaner heuristic for next action """
        return


class NaiveCleaner(Cleaner):
    """
    This naive Cleaner tries to get to the closest dirt left within the map
    """

    def __init__(self, agent_id, agent_max_vel):
        self.curr_goal = None

        Cleaner.__init__(self, agent_id, agent_max_vel)

    def get_next_goal(self):
        if len(self.dirt_pieces) > 0 and self.curr_t is not None:
            if self.curr_goal not in self.dirt_pieces:
                # first goal OR someone picked this dirt, need to change goal
                curr_loc = self.map_service.position_to_map(
                    self.curr_t[::-1][1:])
                closest = None
                min_dist = np.inf
                for dirt in self.dirt_pieces:
                    dist = np.linalg.norm(curr_loc - np.array(dirt))
                    if dist < min_dist:
                        min_dist = dist
                        closest = dirt
                self.curr_goal = closest

                if self.curr_goal is not None:
                    coordinate = self.map_service.map_to_position(
                        np.array([self.curr_goal[1], self.curr_goal[0]]))
                    print('Agent', self.agent_id,
                          'trying to get dirt at', coordinate)
                    self.go_to(coordinate[0], coordinate[1])


class OurCleaner(Cleaner):
    def __init__(self, agent_id, agent_max_vel):
        self.curr_goal = None
        self.clf = None
        self.tie_breaker = None
        self.fig_num = 1

        Cleaner.__init__(self, agent_id, agent_max_vel)

    def get_next_goal(self):
        if len(self.dirt_pieces) > 0 and self.curr_t is not None and self.adversary_t is not None:
            if self.tie_breaker is None:
                self.get_tie_breaker()
                coordinate = self.map_service.map_to_position(
                    np.array([self.tie_breaker[1], self.tie_breaker[0]]))
                print('Agent', self.agent_id,
                      'trying to get to tie break at', coordinate)
                self.go_to(coordinate[0], coordinate[1])

            elif self.tie_breaker not in self.dirt_pieces:
                self.split_environment()
                advantage, on_my_side = self.get_dirt_on_my_side()
                if advantage and len(on_my_side) > 0:
                    curr_loc = self.map_service.position_to_map(
                        self.curr_t[::-1][1:])
                    closest = None
                    min_dist = np.inf
                    for dirt in on_my_side:
                        dist = np.linalg.norm(curr_loc - np.array(dirt))
                        if dist < min_dist:
                            min_dist = dist
                            closest = dirt
                    if self.curr_goal != closest:
                        coordinate = self.map_service.map_to_position(
                            np.array([closest[1], closest[0]]))
                        print('Agent', self.agent_id,
                              'trying to get dirt at', coordinate)
                        self.go_to(coordinate[0], coordinate[1])
                        self.curr_goal = closest
                else:
                    self.tie_breaker = None

    def get_tie_breaker(self):
        my_loc = self.map_service.position_to_map(self.curr_t[::-1][1:])
        dirts = []
        for dirt in self.dirt_pieces:
            dirts.append((dirt, np.linalg.norm(my_loc - np.array(dirt))))
            dirts.sort(key=lambda x: x[1])

        tb_idx = len(dirts) // 2
        self.tie_breaker = dirts[tb_idx][0]

    def split_environment(self):
        # classify dirt pieces based on robots location
        my_loc = self.map_service.position_to_map(
            self.curr_t[::-1][1:]).tolist()
        adversary_loc = self.map_service.position_to_map(
            self.adversary_t[::-1][1:]).tolist()
        X = [my_loc, adversary_loc]
        Y = [self.agent_id, self.adversary_id]
        self.clf = svm.SVC(kernel='linear')
        self.clf.fit(X, Y)

    def get_dirt_on_my_side(self):
        prediction = self.clf.predict(self.dirt_pieces)
        on_my_side = [x for i, x in enumerate(
            self.dirt_pieces) if prediction[i] == self.agent_id]
        advantage = len(on_my_side) >= (len(prediction) - len(on_my_side))
        return advantage, on_my_side

    def save_map(self, on_my_side=None, tie_breaker=None):
        my_color = 'blue'
        adversary_color = 'red'
        plt.figure()
        plt.imshow(self.map_service.map_arr)

        my_loc = self.map_service.position_to_map(
            self.curr_t[::-1][1:]).tolist()
        plt.scatter([my_loc[1]], [my_loc[0]], color=my_color)
        adversary_loc = self.map_service.position_to_map(
            self.adversary_t[::-1][1:]).tolist()
        plt.scatter([adversary_loc[1]], [adversary_loc[0]],
                    color=adversary_color)

        if on_my_side is None:
            for dirt in self.dirt_pieces:
                plt.scatter([dirt[1]], [dirt[0]], marker='*', color='gray')
        else:
            for dirt in self.dirt_pieces:
                if dirt in on_my_side:
                    plt.scatter([dirt[1]], [dirt[0]],
                                marker='*', color=my_color)
                else:
                    plt.scatter([dirt[1]], [dirt[0]], marker='*',
                                color=adversary_color)

        if tie_breaker is not None:
            plt.scatter([tie_breaker[1]], [tie_breaker[0]],
                        marker='*', color='yellow')

        fig = plt.gcf()
        ax = fig.gca()
        plt.savefig(str(self.fig_num)+'.png')
        self.fig_num += 1

########################### Part 2 ##########################
############ Utilities ##################


def probNormalDist(mean, cov, val):
    # Probability of value in normal distribution being under val
    prob = norm(loc=mean, scale=cov).cdf(val)
    return prob


def probNormalDistConfined(mean, cov, val1, val2):
    # Probability of value in normal distribution being between val1 and val2
    prob1 = probNormalDist(mean, cov, val1)
    prob2 = probNormalDist(mean, cov, val2)
    prob = np.abs(prob1 - prob2)
    return prob


def probNormalDistHigher(mean, cov, val):
    # Probability of value in normal distribution being above val
    prob = 1 - probNormalDist(mean, cov, val)
    return prob


def createObsracles(n_obs, n_x_obs, n_y_obs, N):
    # random n_obs obstacles all with width=n_x_obs, height=n_y_obs that fit in the environment with size NxN
    C_obs = []

    while len(C_obs) < n_obs:
        bottom_left_x = np.random.uniform(0, N-n_x_obs)
        bottom_left_y = np.random.uniform(0, N-n_y_obs)
        new_obstacle = Polygon([(bottom_left_x, bottom_left_y),
                                (bottom_left_x, bottom_left_y+n_y_obs),
                                (bottom_left_x+n_x_obs, bottom_left_y+n_y_obs),
                                (bottom_left_x+n_x_obs, bottom_left_y)])

        add = True
        for obs in C_obs:
            if new_obstacle.intersects(obs):
                add = False
                break
        if add:
            C_obs.append(new_obstacle)

    return C_obs


def C_obsPlotter(C_obs):
    # plot obstacles
    for i in range(len(C_obs)):
        x, y = C_obs[i].exterior.xy
        plt.plot(x, y, 'k')


def euclidianDist(pos1, pos2):
    # calculate 2 dimentional euclidian distance
    dist = np.sqrt(((pos1[0] - pos2[0]) ** 2) +
                   ((pos1[1] - pos2[1]) ** 2))
    return dist


def collisionAccured(line, obstacles):
    lineString = LineString(line)
    return lineString.intersects(obstacles)


def multiplyValuesInList(value_list):
    tot_val = value_list[0]
    for i in range(len(value_list) - 1):
        tot_val = tot_val*value_list[i+1]
    return tot_val


################### PRM #####################
class Plotter:
    def __init__(self):
        self.fig = plt.figure(2)
        self.ax = self.fig.subplots()

    def add_obstacles(self, obstacles):
        for obstacle in obstacles:
            self.ax.add_patch(plt.Polygon(
                obstacle.exterior.coords, color='black'))

    def add_prm(self, prm):
        for i, node in enumerate(prm):
            # print('plot',i+1,'/',len(prm))
            for edge in node.edges_lines:
                self.ax.add_line(plt.Line2D(
                    list(edge.coords.xy[0]), list(edge.coords.xy[1]), 1, alpha=0.3))

        # # plot the nodes after the edges so they appear above them
        # for node in prm:
        #     # Do not plot isolated nodes
        #     if len(node.edges) > 0:
        #         plt.scatter(*node.pos.coords.xy, 4, color='red')

    def add_sol(self, sol):
        for i, node in enumerate(sol):
            if node != sol[-1]:
                next_node = sol[i+1]
                for j, edge in enumerate(node.edges):
                    if next_node == edge[0]:
                        line = node.edges_lines[j]
                        self.ax.add_line(plt.Line2D(list(line.coords.xy[0]), list(line.coords.xy[1]),
                                                    1, color='red'))

    def show_prm_graph(self, N_nodes, thd, edges, avg_node_degree):
        plt.autoscale()
        plt.title('$N_{nodes}=$'+str(N_nodes)+', $th_{d}=$'+str(thd)+', #edges=' +
                  str(int(edges))+', avg node degree=%.2f' % (avg_node_degree))
        plt.grid(color='lightgray', linestyle='--')

    def show_sol(self, N_nodes, thd, alg, cost):
        plt.autoscale()
        plt.title('$N_{nodes}=$'+str(N_nodes)+', $th_{d}=$' +
                  str(thd)+', Algorithm='+alg+', path cost=%.2f' % (cost))
        plt.grid(color='lightgray', linestyle='--')
        plt.show()


class NodePRM:
    def __init__(self, pos):
        self.pos = pos
        self.edges = []
        self.edges_lines = []  # hold the lines for the plots

    def add_edge(self, neighbour, edge):
        # every edge in self.edges is described by the neighbour node and the weight
        self.edges.append((neighbour, edge.length))
        self.edges_lines.append(edge)


def GeneratePRM(thd, N_nodes, C_obs, N):
    # input:   thd     - max distance threshold for edge to exists between two nodes
    #          N_nodes - number of nodes
    #          C_obs   - set of obstacles
    #          N       - size of the environment NxN
    # output:  a PRM G=(V,E) as a list of Nodes
    graph = []

    for i in range(N_nodes):
        # draw a sample from the 2D environment
        if i == 0:
            # initial location
            sample = Point(0, 0)
        elif i == 1:
            # goal locarion
            sample = Point(N, N)
        else:
            sample = Point(np.random.uniform(0, N, 2))

        # check if the sample is collision free (in C_free)
        in_c_free = True
        for obstacle in C_obs:
            if sample.intersects(obstacle):
                in_c_free = False
                break

        if in_c_free:
            # add the new node to the graph
            new_vertex = NodePRM(sample)
            graph.append(new_vertex)

            for v in graph:
                if new_vertex == v:
                    continue

                edge = LineString([new_vertex.pos, v.pos])
                # consider all neighbours that are less than thd away
                if edge.length < thd:
                    # check if the edge intersects any obstacle
                    connect = True
                    for obstacle in C_obs:
                        if edge.intersects(obstacle):
                            connect = False
                            break

                    if connect:
                        new_vertex.add_edge(v, edge)
                        v.add_edge(new_vertex, edge)
    return graph


def GetPRMStatistics(prm):
    # Input:    PRM
    # output:   edges - total number of edges in the PRM graph
    #           avg_node_degree - the avg of all nodes degrees
    degrees = 0
    connected_nodes = 0
    for node in prm:
        degrees += len(node.edges)
        # we do not count for isolated nodes in the prm (ofcourse this is only semantic)
        if len(node.edges) > 0:
            connected_nodes += 1

    edges = degrees/2
    avg_node_degree = degrees/connected_nodes

    return edges, avg_node_degree


def prmFindTopRight(prm):
    goal = None
    top_right = Point(N, N)
    min_dist = np.inf

    for node in prm:
        edge = LineString([node.pos, top_right])
        dist = edge.length
        if dist < min_dist:
            min_dist = dist
            goal = node

    return goal


def prmFindBottomLeft(prm):
    start = None
    bottom_left = Point(0, 0)
    min_dist = np.inf

    for node in prm:
        edge = LineString([bottom_left, node.pos])
        dist = edge.length
        if dist < min_dist:
            min_dist = dist
            start = node

    return start


def newPrmSolWithoutOneEdge(prm, node1, node2):

    prm_new = prm
    value_corrected = False

    for i, prm_node in enumerate(prm_new):
        if prm_node == node1:
            for j, prm_edge in enumerate(prm_node.edges):
                if prm_edge[0] == node2:

                    prm_node.edges.pop(j)
                    prm_node.edges_lines.pop(j)
                    prm_new[i] = prm_node
                    value_corrected = True
                    break

        if value_corrected == True:
            break

    value_corrected = False

    for i, prm_node in enumerate(prm_new):
        if prm_node == node2:
            for j, prm_edge in enumerate(prm_node.edges):
                if prm_edge[0] == node1:

                    prm_node.edges.pop(j)
                    prm_node.edges_lines.pop(j)
                    value_corrected = True
                    break

        if value_corrected == True:
            break

    start = prmFindBottomLeft(prm_new)
    goal = prmFindTopRight(prm_new)
    AStar = AStarPlanner(prm_new, start, goal)
    new_sol, new_cost = AStar.Plan()
    return new_sol, new_cost


################ Kalman Filter ######################
class belief:
    def __init__(self, mu, Sigma):
        self.mu = mu
        self.Sigma = Sigma


class KalmanFilter:
    def __init__(self, A, B, C, R, Q, mu_0, Sigma_0, x_0, k):
        self.A = A
        self.B = B
        self.C = C
        self.R = R
        self.Q = Q
        self.x_0 = x_0
        self.belief_0 = belief(mu_0, Sigma_0)
        self.k = k

    def plotBelief(self, t, Belief, cost=0, num_of_help=0):
        mu_x = list()
        mu_y = list()

        for i in range(t):
            mu_val = ((Belief[i]).mu).A
            mu_x.extend(mu_val[0])
            mu_y.extend(mu_val[1])

        plt.plot(mu_x, mu_y, 'g', label="Mean")
        plt.legend()
        plt.xlabel('x - axis')
        plt.ylabel('y - axis')
        # plt.title('Plan Cost: ' + str(round(cost, 2)) +
        #           ', Number of help requasts:' + str(round(num_of_help, 2)))
        plt.grid(color='lightgray', linestyle='--')
        plt.autoscale()

        fignum = plt.gcf().number
        for i in range(t):
            self.drawCovarianceEllipse(
                ((Belief[i]).mu).A, ((Belief[i]).Sigma).A, fignum)

    def plotTrajectoryAndBelief(self, X, t, Belief, cost=0, num_of_help=0):
        x_location = list()
        y_location = list()
        mu_x = list()
        mu_y = list()

        for i in range(t):
            X_t = (X[i]).A
            x_location.extend(X_t[0])
            y_location.extend(X_t[1])

            mu_val = ((Belief[i]).mu).A
            mu_x.extend(mu_val[0])
            mu_y.extend(mu_val[1])

        plt.plot(x_location, y_location, 'r', label="Location")
        plt.plot(mu_x, mu_y, 'g', label="Mean")
        plt.legend()
        plt.xlabel('x - axis')
        plt.ylabel('y - axis')
        # plt.title('Plan Cost: ' + str(round(cost, 2)) +
        #           ', Number of fulfiled help requasts:' + str(round(num_of_help, 2)))
        plt.grid(color='lightgray', linestyle='--')
        plt.autoscale()

        fignum = plt.gcf().number
        for i in range(t):
            self.drawCovarianceEllipse(
                ((Belief[i]).mu).A, ((Belief[i]).Sigma).A, fignum)

    def PropagateUpdateBelief(self, belief_minus_1, u_t, z_t=0, with_observations=False):
        mu_bar_t = self.A * belief_minus_1.mu + self.B * u_t
        Sigma_bar_t = self.A * belief_minus_1.Sigma * self.A.T + self.R

        if with_observations == True:
            K = Sigma_bar_t * self.C.T * (self.C *
                                          Sigma_bar_t * self.C.T + self.Q).I
            mu_t = mu_bar_t + K * (z_t - self.C * mu_bar_t)
            Sigma_t = (np.matrix(np.eye(len(K * self.C))) -
                       K * self.C) * Sigma_bar_t

        else:
            mu_t = mu_bar_t
            Sigma_t = Sigma_bar_t

        belief_t = belief(mu_t, Sigma_t)
        return belief_t

    def SampleMotionModel(self, x, u):
        epsilon_mean = np.transpose(np.zeros(len(x)))
        epsilon = np.matrix(
            np.random.multivariate_normal(epsilon_mean, self.R, 1))
        epsilon = epsilon.T
        x_next = self.A * x + self.B * u + epsilon
        return x_next

    def drawCovarianceEllipse(self, mu, Sigma, fignum):  # needs more input
        fig = plt.figure(fignum)
        axes = fig.gca()
        w, v = np.linalg.eig(Sigma)
        angle = np.arctan2(v[1, 0], v[0, 0])

        e1 = pat.Ellipse(mu, np.sqrt(
            w[0] * self.k), np.sqrt(w[1] * self.k), np.rad2deg(angle), fill=False)
        e1.set_edgecolor('b')
        e1.set_linestyle('--')
        axes.add_patch(e1)

    def cheakObstaclesInCovariance(self, belief, maze):
        w, v = np.linalg.eig(belief.Sigma)
        angle = np.arctan2(v[1, 0], v[0, 0])

        e1 = shapely_ellipse(belief.mu,
                                          np.sqrt(w[0] * self.k),
                                          np.sqrt(w[1] * self.k),
                                          np.rad2deg(angle))

        if e1.area > 0:
            cross = maze.intersection(e1)
            if cross.area > 0:
                prob_x = probNormalDistConfined(
                    belief.mu[0], belief.Sigma[0, 0],
                    cross.bounds[0], cross.bounds[2])
                prob_y = probNormalDistConfined(
                    belief.mu[1], belief.Sigma[1, 1],
                    cross.bounds[1], cross.bounds[3])
                prob = prob_x * prob_y
                return np.float64(prob)

        return 0


    def maze(num_obstacales, circle_buffer, wall_buffer, max_x, min_x, max_y, min_y):
        Circles_obs = []

        for i in range(num_obstacales):
            x_center = random.randint(min_x + 1, max_x - 1)
            y_center = random.randint(min_y + 1, max_y - 1)
            circle_obs = Point(x_center, y_center).buffer(circle_buffer)
            Circles_obs.append(circle_obs)

            x, y = circle_obs.exterior.xy
            plt.plot(x, y, 'k')

        maze_wall_in = [(min_x, min_y), (min_x, max_y),
                        (max_x, max_y), (max_x, min_y)]
        maze_wall_out = [(min_x - wall_buffer, min_y - wall_buffer),
                         (min_x - wall_buffer, max_y + wall_buffer),
                         (max_x + wall_buffer, max_y + wall_buffer),
                         (max_x + wall_buffer, min_y - wall_buffer)]
        maze_wall = Polygon(maze_wall_out, [maze_wall_in])

        x, y = maze_wall.exterior.xy
        plt.plot(x, y, 'k')
        x, y = maze_wall.interiors[0].xy
        plt.plot(x, y, 'k')

        maze_obstacales = maze_wall
        for i in range(num_obstacales):
            maze_obstacales = maze_obstacales.union(Circles_obs[i])

        return maze_obstacales

    def apriorySimWithHelpSteps(self, maze, action_speed, C_obs, sol, cost, help_steps=list(), print_plt=False):
        route_points = initializeAction(sol)
        Belief = [self.belief_0]
        steps_p_colide = list()
        num_of_steps = 0
        num_of_help = 0

        for k in range(len(route_points) - 1):
            curr_pos = (((Belief[-1]).mu).A1).tolist()
            next_pos = route_points[k+1]
            U = actionsWithPositions(curr_pos, next_pos, action_speed)

            for i in range(len(U)):
                belief_new = self.PropagateUpdateBelief(Belief[-1], U[i])

                if len(help_steps) > num_of_help and help_steps[num_of_help] == num_of_steps:
                    step_p_colide = self.cheakObstaclesInCovariance(
                        Belief[-1], maze)
                    steps_p_colide.append(step_p_colide)
                    num_of_help += 1
                    Belief[-1] = belief(Belief[-1].mu, self.belief_0.Sigma)
                    belief_new = self.PropagateUpdateBelief(
                        Belief[-1], U[i])

                else:
                    Belief.append(belief_new)
                    num_of_steps += 1

        if print_plt == True:
            plotPlannedRoute(route_points)
            C_obsPlotter(C_obs)
            self.plotBelief(num_of_steps, Belief, cost, num_of_help)
            plt.show()

        if len(steps_p_colide) > 0:
            steps_p_not_colide = [1 - p for p in steps_p_colide]
            p_colide = 1 - multiplyValuesInList(steps_p_not_colide)
            return p_colide
        else:
            return 0

    def apriorySimWithCourseCorrection(self, maze, action_speed, C_obs, sol, cost, p_colide_threshold, helper, print_plt=False):
        route_points = initializeAction(sol)
        Belief = [self.belief_0]
        num_of_steps = 0
        num_of_help = 0
        max_p_collide = 0
        help_positions = list()
        help_steps = list()

        for k in range(len(route_points) - 1):
            curr_pos = (((Belief[-1]).mu).A1).tolist()
            next_pos = route_points[k+1]
            U = actionsWithPositions(curr_pos, next_pos, action_speed)

            for i in range(len(U)):
                belief_new = self.PropagateUpdateBelief(Belief[-1], U[i])
                p_colide = self.cheakObstaclesInCovariance(Belief[-1], maze)
                max_p_collide = np.max([max_p_collide, p_colide])

                if p_colide > p_colide_threshold:
                    num_of_help += 1
                    Belief[-1] = belief(Belief[-1].mu, self.belief_0.Sigma)
                    belief_new = self.PropagateUpdateBelief(
                        Belief[-1], U[i])
                    help_pos = (((Belief[-1]).mu).A1).tolist()
                    help_positions.append(help_pos)
                    help_steps.append(num_of_steps)

                    if print_plt == True:
                        print('Location of help request [',
                              round(help_pos[0], 1), ',',
                              round(help_pos[1], 1), ']')
                        print('Estimated max probability of colliding without information in the next step: ' +
                              str(round(p_colide + 0.0026, 4)) + '\n')

                else:
                    Belief.append(belief_new)
                    num_of_steps += 1

        cost_of_help = 0
        for help_pos in help_positions:
            cost_of_help += helper.assistCost(help_pos)
            helper.newPos(help_pos)

        if print_plt == True:
            plotPlannedRoute(route_points)
            C_obsPlotter(C_obs)
            self.plotBelief(num_of_steps, Belief, cost, num_of_help)
            helper.plotHelper()
            plt.show()

        return help_steps, cost_of_help

    def runSimWithCourseCorrection(self, action_speed, C_obs, sol, cost, helper, maze, helps=[], print_plt=False):
        route_points = initializeAction(sol)
        X = [self.x_0]
        Belief = [self.belief_0]
        num_of_steps = 0
        currection_steps = 0
        num_of_help = 0

        for k in range(len(route_points) - 1):
            curr_pos = (((Belief[-1]).mu).A1).tolist()
            next_pos = route_points[k+1]
            U = actionsWithPositions(curr_pos, next_pos, action_speed)

            for i in range(len(U)):
                x_new = self.SampleMotionModel(X[-1], U[i])
                belief_new = self.PropagateUpdateBelief(Belief[-1], U[i])

                if len(helps) > num_of_help and helps[num_of_help] == num_of_steps:
                    X.pop()
                    Belief.pop()
                    i -= 1
                    num_of_steps -= 1
                    num_of_help += 1

                    curr_pos = ((X[-1]).A1).tolist()
                    next_pos = ((Belief[-1]).mu.A1).tolist()
                    U_correction = actionsWithPositions(
                        curr_pos, next_pos, float(action_speed)/4)

                    belief_prev = belief(X[-1], self.belief_0.Sigma)
                    Belief[-1] = belief_prev

                    for j in range(len(U_correction)):
                        x_new = self.SampleMotionModel(X[-1], U_correction[j])
                        belief_new = self.PropagateUpdateBelief(
                            Belief[-1], U_correction[j])
                        X.append(x_new)
                        Belief.append(belief_new)
                        currection_steps += 1

                else:
                    X.append(x_new)
                    Belief.append(belief_new)
                    num_of_steps += 1

        if print_plt == True:
            plotPlannedRoute(route_points)
            C_obsPlotter(C_obs)
            self.plotTrajectoryAndBelief(
                X, currection_steps + num_of_steps, Belief, cost, num_of_help)
            helper.plotHelper()
            plt.show()

        # TODO: maybe change into how many collisions instead of how if a collision accured
        X_list = list()
        for x in X:
            X_list.append(x.A1)
        lineString = LineString(X)
        return lineString.intersects(maze)

def shapely_ellipse(center, major_axis, minor_axis, rotation=0.):
        el = Point(0., 0.).buffer(1.)
        el = sa.scale(el, major_axis/2, minor_axis/2)
        el = sa.rotate(el, rotation)
        el = sa.translate(el, *center)
        return el

def actionsWithPositions(curr_pos, next_pos, action_speed):
    actions = []

    deg = np.arctan2(
        (next_pos[1] - curr_pos[1]),
        (next_pos[0] - curr_pos[0]))
    x_speed = action_speed * np.cos(deg)
    y_speed = action_speed * np.sin(deg)
    u = np.matrix([[x_speed], [y_speed]])

    distance = np.sqrt(
        (next_pos[0] - curr_pos[0]) ** 2 +
        (next_pos[1] - curr_pos[1]) ** 2)
    num_of_actions = int(math.ceil(distance/action_speed))
    for j in range(num_of_actions):
        actions.append(u)

    return actions


def initializeAction(sol):
    resulution = 5
    route_points = []
    full_route_points = []

    for i in range(len(sol)):
        point_location = [sol[-i-1].pos.bounds[0],
                          sol[-i-1].pos.bounds[1]]
        route_points.append(point_location)

        if 0 < i:
            prev_point = route_points[i - 1]
            next_point = route_points[i]
            one_step = [(next_point[0] - prev_point[0])/resulution,
                        (next_point[1] - prev_point[1])/resulution]

            for j in range(resulution):
                curr_point = [prev_point[0] + j * one_step[0],
                              prev_point[1] + j * one_step[1]]
                full_route_points.append(curr_point)

    full_route_points.append(point_location)

    return full_route_points


def plotPlannedRoute(route_points):
    x = []
    y = []

    for i in range(len(route_points)):
        xy = route_points[i]
        x.append(xy[0])
        y.append(xy[1])

    plt.plot(x, y, ':m', label="Plan")

################ Helper Agent ##################


class Helper:
    def __init__(self, pos, speed):
        self.pos = [pos]
        self.speed = speed

    def assistPossible(self, new_pos, time_limit):
        dist = euclidianDist(self.pos[-1], new_pos)
        return time_limit * self.speed > dist

    def assistCost(self, new_pos):
        return euclidianDist(self.pos[-1], new_pos)

    def newPos(self, new_pos):
        self.pos.append(new_pos)

    def possibleHelpPoints(self, num_of_help, help_steps, help_positions, asker_action_speed):
        prev_help_time = 0
        tot_cost_of_help = 0
        possible_help_steps = []

        for j in range(num_of_help):
            time_limit = help_steps[j] * asker_action_speed - prev_help_time

            if self.assistPossible(help_positions[j], time_limit):
                tot_cost_of_help += self.assistCost(help_positions[j])
                self.newPos(help_positions[j])
                prev_help_time = time_limit
                possible_help_steps.append(help_steps[j])

            else:
                print('help not possible in location[',
                      round((help_positions[j])[0], 1), ',',
                      round((help_positions[j])[1], 1), ']')

        return self, tot_cost_of_help, possible_help_steps

    def plotHelper(self):
        helper_pos_x = []
        helper_pos_y = []

        for curr_pos in self.pos:
            helper_pos_x.append(curr_pos[0])
            helper_pos_y.append(curr_pos[1])

        plt.plot(helper_pos_x, helper_pos_y, 'kp:', label="Helper")
        plt.legend()
        plt.xlabel('x - axis')
        plt.ylabel('y - axis')
        plt.grid(color='lightgray', linestyle='--')
        plt.autoscale()

################ A* ###########################


class NodeAStar:
    """
    Helper class to hold the nodes of Astar.
    A node keeps track of potentially a parent node.
    A node has its cost (g_score , the cost of the cheapest path from start to it),
    and the f-score (cost + heuristic to goal).
    """

    def __init__(self, prm_node,
                 parent_node=None,
                 g_score=0,
                 f_score=None):

        self.prm_node = prm_node
        self.parent_node = parent_node
        self.g_score = g_score
        self.f_score = f_score

        if self.parent_node is not None:
            self.g_score += self.parent_node.g_score

    def __lt__(self, other):
        return self.f_score < other.f_score


class AStarPlanner(object):
    def __init__(self, prm, start, goal):
        self.start = start
        self.goal = goal
        self.prm = prm
        self.nodes = dict()

        self.open = AstarPriorityQueue()
        self.close = AstarPriorityQueue()

    def Plan(self):
        plan = []

        start_node = NodeAStar(self.start, None, 0)
        start_node.f_score = self._calc_node_f_score(start_node)
        self.open.Insert(start_node, start_node.f_score)

        count = 0
        while not self.open.IsEmpty():
            next_node = self.open.Pop()
            self.close.Insert(next_node, 1)  # priority has no meaning in close
            self.nodes[next_node.prm_node] = next_node
            plan.append(next_node.prm_node)
            count += 1

            if next_node.prm_node == self.goal:
                break

            edges = self._get_neighbours(next_node)
            for edge in edges:
                neighbour, cost = edge
                successor_node = NodeAStar(neighbour, next_node, cost)
                successor_node.f_score = self._calc_node_f_score(
                    successor_node)

                new_g = successor_node.g_score
                # the node is already in OPEN
                if self.open.Contains(successor_node.prm_node):
                    already_found_node = self.open.GetByPRMNode(
                        successor_node.prm_node)
                    if new_g < already_found_node.g_score:  # new parent is better
                        already_found_node.g_score = new_g
                        already_found_node.parent_node = successor_node.parent_node
                        already_found_node.f_score = self._calc_node_f_score(
                            already_found_node)

                        # f changed so need to reposition in OPEN
                        self.open.Remove(already_found_node.prm_node)
                        self.open.Insert(already_found_node,
                                         already_found_node.f_score)

                    else:  # old path is better - do nothing
                        pass
                else:  # state not in OPEN maybe in CLOSED
                    # this node exists in CLOSED
                    if self.close.Contains(successor_node.prm_node):
                        already_found_node = self.close.GetByPRMNode(
                            successor_node.prm_node)
                        if new_g < already_found_node.g_score:  # new parent is better
                            already_found_node.g_score = new_g
                            already_found_node.parent_node = successor_node.parent_node
                            already_found_node.f_score = self._calc_node_f_score(
                                already_found_node)

                            # move old node from CLOSED to OPEN
                            self.close.Remove(already_found_node.prm_node)
                            self.nodes.pop(already_found_node.prm_node)
                            self.open.Insert(
                                already_found_node, already_found_node.f_score)
                        else:  # old path is better - do nothing
                            pass
                    else:
                        # this is a new state - create a new node = insert new node to OPEN
                        self.open.Insert(
                            successor_node, successor_node.f_score)

        # print("Astar expanded", count, "nodes")

        return self._backtrace(plan)

    def _backtrace(self, plan):
        """
        backtrace from goal to start
        """
        cost = 0
        current = self.nodes[plan[-1]]
        sol = [current.prm_node]
        while current.prm_node != plan[0]:
            cost += self._get_distance_between_nodes(
                current, current.parent_node)
            current = self.nodes[current.parent_node.prm_node]
            sol.append(current.prm_node)

        return sol, cost

    def _calc_node_f_score(self, node):
        return node.g_score + self._compute_heuristic(node)

    def _compute_heuristic(self, node):
        """
        Heuristic is defined as Euclidean distance to goal
        """
        edge = LineString([node.prm_node.pos, self.goal.pos])
        return edge.length

    def _get_neighbours(self, node):
        """
        Returns the edges in the PRM
        each edge is a tuple of (neighbour, cost)
        """
        return node.prm_node.edges

    def _get_distance_between_nodes(self, node1, node2):
        """
        Returns Euclidean distance between two nodes in the graph
        """
        edge = LineString([node1.prm_node.pos, node2.prm_node.pos])
        return edge.length


class AstarPriorityQueue:
    def __init__(self):
        self.elements = []
        # just for performance (could probably used ordered_dict..)
        self.elements_dict = {}

    def IsEmpty(self):
        return len(self.elements) == 0

    def Insert(self, item, priority):
        heapq.heappush(self.elements, (priority, item))
        self.elements_dict[item.prm_node] = item

    def Pop(self):
        item = heapq.heappop(self.elements)
        self.elements_dict.pop(item[1].prm_node)
        return item[1]

    def TopKey(self):
        return heapq.nsmallest(1, self.elements)[0][0]

    def Remove(self, prm_node):
        self.elements = [e for e in self.elements if e[1].prm_node != prm_node]
        heapq.heapify(self.elements)

        self.elements_dict.pop(prm_node)

    def Contains(self, prm_node):
        return prm_node in self.elements_dict

    def GetByPRMNode(self, prm_node):
        return self.elements_dict[prm_node]

################# Main functions ####################


def createPath():
    # create obstacles
    size_env = 100
    n_obs = 30
    n_x_obs = 5
    n_y_obs = 5
    C_obs = createObsracles(n_obs, n_x_obs, n_y_obs, size_env)

    # calculate the prm
    N_nodes = 100
    thd = 50

    prm = GeneratePRM(thd, N_nodes, C_obs, size_env)
    start = prmFindBottomLeft(prm)
    goal = prmFindTopRight(prm)

    # Solve Using AStar
    AStar = AStarPlanner(prm, start, goal)
    sol, cost = AStar.Plan()

    if start not in sol or goal not in sol:
        connected = False
    else:
        connected = True

    return sol, cost, C_obs, start, connected


def calcAndEstimateCollisionsWithHelp(sol, cost, C_obs, start, n_tests, p_colide_threshold, print_results):
    # initialization Kalman Filter
    A = np.matrix('1 0; 0 1')
    B = np.matrix('1 0; 0 1')
    C = np.matrix('1 0; 0 1')
    R = np.matrix('0.01 0; 0 0.01')
    Q = np.matrix('0.01 0; 0 0.01')
    x_0 = np.matrix([[start.pos.bounds[0]], [start.pos.bounds[1]]])
    Sigma_0 = np.matrix('0 0 ; 0 0')
    mu_0 = x_0
    k = 11.82  # value for 3Sigma meaning 99.74% probability
    KF = KalmanFilter(A, B, C, R, Q, mu_0, Sigma_0, x_0, k)

    # initializing maze
    maze = C_obs[0]
    for i in range(len(C_obs)):
        maze = maze.union(C_obs[i])

    # create helper agent
    helper_init_pos = [0, 0]
    helper_speed = 2
    helper = Helper(helper_init_pos, helper_speed)

    # run simulation to examin where help is needed
    action_speed = 1
    help_steps, cost_of_help = KF.apriorySimWithCourseCorrection(
        maze, action_speed, C_obs, sol, cost, p_colide_threshold, helper, print_plt=print_results)

    collideWithHelp = 0
    for i in range(n_tests):
        # calc probability of collide with help
        collideWithHelp += KF.runSimWithCourseCorrection(
            action_speed, C_obs, sol, cost, helper, maze, help_steps)

    if print_results == True:
        # print plot with help
        KF.runSimWithCourseCorrection(
            action_speed, C_obs, sol, cost, helper, maze, help_steps, print_plt=print_results)
        # calculate maximal probability of colliding without information
        KF.apriorySimWithHelpSteps(
            maze, action_speed, C_obs, sol, cost, print_plt=print_results)

    return collideWithHelp/n_tests, cost_of_help


def vacuum_cleaning(agent_id, agent_max_vel, adversary=0):
    ourCleaner = OurCleaner(agent_id, agent_max_vel)
    if adversary:
        adversary_id = 0 if agent_id == 1 else 1
        naiveCleaner = NaiveCleaner(adversary_id, agent_max_vel)


def inspection(agent_id, agent_max_vel):
    print('start inspection')
    connected = False
    while connected == False:
        sol, cost, C_obs, start, connected = createPath()

    test_with_help, cost_of_help = calcAndEstimateCollisionsWithHelp(
        sol, cost, C_obs, start, n_tests=1, p_colide_threshold=0, print_results=True)


# If the python node is executed as main process (sourced directly)
if __name__ == '__main__':

    # Initializes a rospy node to let the SimpleActionClient publish and subscribe
    rospy.init_node('assignment3')

    exec_mode = sys.argv[1]
    print('exec_mode:' + exec_mode)

    if exec_mode == 'cleaning':
        agent_id = int(sys.argv[2])
        print('agent id:', agent_id)
        agent_max_vel = eval(sys.argv[3])
        adversary = 0
        if len(sys.argv) > 4:
            adversary = int(sys.argv[4])
        vacuum_cleaning(agent_id, agent_max_vel, adversary)

    elif exec_mode == 'inspection':
        agent_id = sys.argv[2]
        agent_max_vel = sys.argv[3]
        inspection(agent_id, agent_max_vel)

    else:
        print("Code not found")
        raise NotImplementedError
