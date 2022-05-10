#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import random
import math
import sys
from collections import defaultdict, deque, Counter
from itertools import combinations
import heapq

class Problem(object):
    """The abstract class for a formal problem. A new domain subclasses this,
    overriding `actions` and `results`, and perhaps other methods.
    The default heuristic is 0 and the default action cost is 1 for all states.
    When yiou create an instance of a subclass, specify `initial`, and `goal` states 
    (or give an `is_goal` method) and perhaps other keyword args for the subclass."""

    def __init__(self, initial=None, goal=None, **kwds): 
        self.__dict__.update(initial=initial, goal=goal, **kwds) 
        
    def actions(self, state):        
        raise NotImplementedError
    def result(self, state, action): 
        raise NotImplementedError
    def is_goal(self, state):        
        return state == self.goal
    def action_cost(self, s, a, s1): 
        return 1
    
    def __str__(self):
        return '{0}({1}, {2})'.format(
            type(self).__name__, self.initial, self.goal)
            
class Node:
    "A Node in a search tree."
    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.__dict__.update(state=state, parent=parent, action=action, path_cost=path_cost)

    def __str__(self): 
        return '<{0}>'.format(self.state)
    def __len__(self): 
        return 0 if self.parent is None else (1 + len(self.parent))
    def __lt__(self, other): 
        return self.path_cost < other.path_cost
        
failure = Node('failure', path_cost=math.inf)
cutoff  = Node('cutoff',  path_cost=math.inf)

def expand(problem, node):
    "Expand a node, generating the children nodes."
    s = node.state
    for action in problem.actions(s):
        s1 = problem.result(s, action)
        cost = node.path_cost + problem.action_cost(s, action, s1)
        yield Node(s1, node, action, cost)
        

def path_actions(node):
    "The sequence of actions to get to this node."
    if node.parent is None:
        return []  
    return path_actions(node.parent) + [node.action]


def path_states(node):
    "The sequence of states to get to this node."
    if node in (cutoff, failure, None): 
        return []
    return path_states(node.parent) + [node.state]
    
class PriorityQueue:
    """A queue in which the item with minimum f(item) is always popped first."""

    def __init__(self, items=(), key=lambda x: x): 
        self.key = key
        self.items = [] # a heap of (score, item) pairs
        for item in items:
            self.add(item)
         
    def add(self, item):
        """Add item to the queuez."""
        pair = (self.key(item), item)
        heapq.heappush(self.items, pair)

    def pop(self):
        """Pop and return the item with min f(item) value."""
        return heapq.heappop(self.items)[1]
    
    def top(self): return self.items[0][1]

    def __len__(self): return len(self.items)
    
def best_first_search(problem, f):
    "Search nodes with minimum f(node) value first."
    node = Node(problem.initial)
    frontier = PriorityQueue([node], key=f)
    reached = {problem.initial: node}
    while frontier:
        node = frontier.pop()
        if problem.is_goal(node.state):
            return node
        for child in expand(problem, node):
            s = child.state
            if s not in reached or child.path_cost < reached[s].path_cost:
                reached[s] = child
                frontier.add(child)
    return failure

def g(n): 
    return n.path_cost
    cost = 1
    return cost
    
class RouteProblem(Problem):
    """A problem to find a route between locations on a `Map`.
    Create a problem with RouteProblem(start, goal, map=Map(...)}).
    States are the vertexes in the Map graph; actions are destination states."""
    
    def actions(self, state): 
        """The places neighboring `state`."""
        return self.map.neighbors[state]
    
    def result(self, state, action):
        """Go to the `action` place, if the map says that is possible."""
        return action if action in self.map.neighbors[state] else state
    
    def action_cost(self, s, action, s1):
        """The distance (cost) to go from s to s1."""
        return self.map.distances[s, s1]
    
    def h(self, node):
        "Straight-line distance between state and the goal."
        locs = self.map.locations
        return straight_line_distance(locs[node.state], locs[self.goal])
        
class Map:
    """A map of places in a 2D world: a graph with vertexes and links between them. 
    In `Map(links, locations)`, `links` can be either [(v1, v2)...] pairs, 
    or a {(v1, v2): distance...} dict. Optional `locations` can be {v1: (x, y)} 
    If `directed=False` then for every (v1, v2) link, we add a (v2, v1) link."""

    def __init__(self, links, locations=None, directed=False):
        if not hasattr(links, 'items'): # Distances are 1 by default
            links = {link: 1 for link in links}
        if not directed:
            for (v1, v2) in list(links):
                links[v2, v1] = links[v1, v2]
        self.distances = links
        self.neighbors = multimap(links)
        self.locations = locations or defaultdict(lambda: (0, 0))

        
def multimap(pairs) -> dict:
    "Given (key, val) pairs, make a dict of {key: [val,...]}."
    result = defaultdict(list)
    for key, val in pairs:
        result[key].append(val)
    return result
    
    
saveetha_nearby_locations = Map(
    {('PERUNGALATHUR', 'TAMBARAM'):  3, ('TAMBARAM', 'CHROMRPET'): 7, ('TAMBARAM', 'THANDALAM'): 10,
     ('CHROMRPET', 'MEDAVAKAM'): 10, ('CHROMRPET', 'THORAIPAKKAM'): 12, ('CHROMRPET', 'GUINDY'): 13, 
     ('MEDAVAKAM', 'SIRUSERI'):  11, ('SIRUSERI', 'KELAMBAKKAM'): 8, ('KELAMBAKKAM', 'THORAIPAKKAM'): 17, 
     ('KELAMBAKKAM', 'VGP'): 18, ('VGP', 'THIRUVALLUVAR'): 8, ('THIRUVALLUVAR', 'ADYAR'):  5, ('ADYAR', 'GUINDY'): 5, 
     ('GUINDY', 'THORAIPAKKAM'): 9, ('GUINDY', 'T-NAGAR'): 5, ('T-NAGAR','MARINABEACH'): 6, ('T-NAGAR','KOYAMBEDU'): 9, 
     ('GUINDY','PORUR'): 10, ('KOYAMBEDU','AMBATTUR'): 10, ('AMBATTUR','AVADI'): 10, ('AVADI','POONAMALLEE'): 9, 
     ('THANDALAM','SAVEETHAENGINEERINGCOLLEGE'): 18, ('SAVEETHAENGINEERINGCOLLEGE','POONAMALLEE'): 10, 
     ('POONAMALLEE','PORUR'): 7, ('THANDALAM','PORUR'): 7})


r0 = RouteProblem('PERUNGALATHUR', 'KELAMBAKKAM', map=saveetha_nearby_locations)
r1 = RouteProblem('PERUNGALATHUR', 'MARINABEACH', map=saveetha_nearby_locations)
r2 = RouteProblem('MARINABEACH', 'SAVEETHAENGINEERINGCOLLEGE', map=saveetha_nearby_locations)
r3 = RouteProblem('SAVEETHAENGINEERINGCOLLEGE', 'VGP', map=saveetha_nearby_locations)
r4 = RouteProblem('TAMBARAM', 'T-NAGAR', map=saveetha_nearby_locations)
r5 = RouteProblem('KOYAMBEDU', 'POONAMALLEE', map=saveetha_nearby_locations)
r6 = RouteProblem('KELAMBAKKAM', 'KOYAMBEDU', map=saveetha_nearby_locations)
r7 = RouteProblem('THIRUVALLUVAR', 'PERUNGALATHUR', map=saveetha_nearby_locations)
r8 = RouteProblem('KELAMBAKKAM', 'SAVEETHAENGINEERINGCOLLEGE', map=saveetha_nearby_locations)
r9 = RouteProblem('CHROMRPET', 'AVADI', map=saveetha_nearby_locations)
print(r0)
print(r1)
print(r2)
print(r3)
print(r4)
print(r5)
print(r6)
print(r7)
print(r8)
print(r9)


# In[2]:


goal_state_path=best_first_search(r0,g)


# In[3]:


print("GoalStateWithPath:{0}".format(goal_state_path))


# In[4]:


path_states(goal_state_path) 


# In[5]:


print("Total Distance={0} Kilometers".format(goal_state_path.path_cost))


# In[6]:


goal_state_path=best_first_search(r1,g)
print("GoalStateWithPath:{0}".format(goal_state_path))


# In[7]:


path_states(goal_state_path) 


# In[8]:


print("Total Distance={0} Kilometers".format(goal_state_path.path_cost))


# In[9]:


goal_state_path=best_first_search(r2,g)
print("GoalStateWithPath:{0}".format(goal_state_path))


# In[10]:


path_states(goal_state_path) 


# In[11]:


print("Total Distance={0} Kilometers".format(goal_state_path.path_cost))


# In[12]:


goal_state_path=best_first_search(r3,g)
print("GoalStateWithPath:{0}".format(goal_state_path))


# In[13]:


path_states(goal_state_path) 


# In[14]:


print("Total Distance={0} Kilometers".format(goal_state_path.path_cost))


# In[15]:


goal_state_path=best_first_search(r4,g)
print("GoalStateWithPath:{0}".format(goal_state_path))


# In[16]:


path_states(goal_state_path) 


# In[17]:


print("Total Distance={0} Kilometers".format(goal_state_path.path_cost))


# In[18]:


goal_state_path=best_first_search(r5,g)
print("GoalStateWithPath:{0}".format(goal_state_path))


# In[19]:


path_states(goal_state_path) 


# In[20]:


print("Total Distance={0} Kilometers".format(goal_state_path.path_cost))


# In[21]:


goal_state_path=best_first_search(r6,g)
print("GoalStateWithPath:{0}".format(goal_state_path))


# In[22]:


path_states(goal_state_path) 


# In[23]:


print("Total Distance={0} Kilometers".format(goal_state_path.path_cost))


# In[24]:


goal_state_path=best_first_search(r7,g)
print("GoalStateWithPath:{0}".format(goal_state_path))


# In[25]:


path_states(goal_state_path) 


# In[26]:


print("Total Distance={0} Kilometers".format(goal_state_path.path_cost))


# In[27]:


goal_state_path=best_first_search(r8,g)
print("GoalStateWithPath:{0}".format(goal_state_path))


# In[28]:


path_states(goal_state_path) 


# In[29]:


print("Total Distance={0} Kilometers".format(goal_state_path.path_cost))


# In[30]:


goal_state_path=best_first_search(r9,g)
print("GoalStateWithPath:{0}".format(goal_state_path))


# In[31]:


path_states(goal_state_path) 


# In[32]:


print("Total Distance={0} Kilometers".format(goal_state_path.path_cost))


# In[ ]:




