import numpy as np
import matplotlib.pyplot as plt
from numpy.random import *

#Temporal Difference Learning
class TD:
    def __init__(self):

        #Initialize Parameters
        self.T = 24 #Time period
        self.D = 60 #Period of time(min)
        self.epsilon = 0.5
        
        #V = np.zeros(24)
        self.V = np.zeros(T)
        self.e = np.zeros(T)
        
        self.alpha = 0.5 #learning rate
        self.gamma = 0.5#discount rate
        self.ld = 0.3 #lambda
        seed(200)
        self.P_charge = np.random.uniform(0.0,1000,24)
        self.P_discharge = np.random.uniform(0.0,1000,24)

        #actions = {Pcharge or Pdischarge}
        self.actions = [[0,1],[1,0]]
        
        #Battery Amortized Cost Constants
        self.Pi_P = 100
        self.Pi_E =  175
        
        #Battery Chemistry Specifications
        self.eta = 0.85
        
        self.P_cap = 300 #random data
        self.E_cap = 600 #random data
        
        self.E_init = 0.30 * self.E_cap #necessary initial charge
        self.E_st = np.zeros(T)
        
        self.P_load = np.array([784.312500 ,735.29296 ,740.250000 ,686.894824 ,689.882812 ,710.783203 ,822.656250 ,971.191406 ,1085.449219 ,1150.03125 ,1171.8000 ,1164.770508 ,1161.084375 ,1164.770508 ,1148.437500 ,1148.437500 ,1145.244141 ,1165.260938 ,1172.812500 ,1149.873047 ,1126.406250 ,1102.500000 ,1054.096875 ,937.500000])

        self.P_pv = np.array([0,0,0, 0, 0, 100, 150, 200, 300, 300, 300, 400, 400, 400, 400, 300, 200, 100 , 0, 0 , 0 ,0, 0, 0])
        
        self.P_grid = np.zeros(T)
    
        self.Price = np.array([0, 10, 10, 10 ,10 ,40, 40 ,40 ,40 ,30 ,20 ,10 ,10 ,30 ,40 ,10 ,20 ,10, 20 ,30 ,40 ,30,10 ,10])
        
        self.Qvalues ={}
        self.Evalues = {}

        
    #Reward Part
    def get_reward(self,state, action,t):
        if action[0] == 1:
            self.P_discharge[t] = 0.0
        elif action[1] == 1:
            self.P_charge[t] = 0.0
        
        self.P_grid[t] = state[0] + self.P_charge[t] - self.P_discharge[t]
    
        #cost = price * max{0, P_grid[t]}
        return state[2] * max(0,self.P_grid[t])

    #Qvalue Part
    def get_Qvalue(self,state, action):
        return self.Qvalues.get((str(state),str(action)), 0.0)
    
    def get_baseline(self):
        return np.sum(np.multiply(self.Price, self.P_load))
    
    def update_Qvalue(self,pstate, action, reward, state, delta):
        #Q(s,a) <- Q(s,a) + alpha * delta * e(s,a)
        oldv = self.Qvalues.get((str(pstate), str(action)), None)
        
        if oldv is None:
            self.Qvalues[(str(pstate),str(action))] = reward
        else:
            #self.Qvalues[(str(pstate), str(action))] = oldv + self.alpha * TD.get_Evalues(self,pstate, action) * delta
            self.Qvalues[(str(pstate), str(action))] = oldv + self.alpha * (reward + TD.get_Evalues(self,state, action) * delta - oldv)

    def get_Evalues(self,state, action):
        return self.Evalues.get((str(state), str(action)),0.0)

    def update_Evalues(self,state, action):
        #e(s) <- gamma * lambda * e(s,a)
        self.Evalues[(str(state), str(action))] = self.gamma * self.ld * TD.get_Evalues(self,state, action)

    #Action Part
    def choose_action(self,state,t):
        #Epsilon-Greedy 
        if np.random.random() > self.epsilon:
            #Choose action randomly
            #return self.actions[random.choice(len(self.actions))]
            return self.actions[np.random.choice(len(self.actions))]
        
        else:
            q = [TD.get_Qvalue(self,state, a) for a in self.actions]
            maxQ = max(q)
            count = q.count(maxQ)
            
            if count > 1:
                best = [i for i in range(len(self.actions)) if q[i] == maxQ]
                #i = random.choice(best)
                i = best[np.random.choice(len(best))]
            else:
                i = q.index(maxQ)
                
            action = self.actions[i]
            return action

class QLearning:
    def __init__(self):
        #Initialize Parameters
        self.T = 24 #Time period
        self.D = 60 #Period of time(min)
        self.epsilon = 0.5
        
        #V = np.zeros(24)
        self.V = np.zeros(T)
        self.e = np.zeros(T)
        
        self.alpha = 0.5 #learning rate
        self.gamma = 0.5#discount rate
        self.ld = 0.3 #lambda

        seed(200)
        self.P_charge = np.random.uniform(0.0,10000,24)
        self.P_discharge = np.random.uniform(0.0,10000,24)
        
        #actions = {Pcharge or Pdischarge}
        self.actions = [[0,1],[1,0]]
        
        #Battery Amortized Cost Constants
        self.Pi_P = 100
        self.Pi_E =  175
        
        #Battery Chemistry Specifications
        self.eta = 0.85
        
        self.P_cap = 300 #random data
        self.E_cap = 600 #random data
        
        self.E_init = 0.30 * self.E_cap #necessary initial charge
        self.E_st = np.zeros(T)
        
        self.P_load = np.array([784.312500 ,735.29296 ,740.250000 ,686.894824 ,689.882812 ,710.783203 ,822.656250 ,971.191406 ,1085.449219 ,1150.03125 ,1171.8000 ,1164.770508 ,1161.084375 ,1164.770508 ,1148.437500 ,1148.437500 ,1145.244141 ,1165.260938 ,1172.812500 ,1149.873047 ,1126.406250 ,1102.500000 ,1054.096875 ,937.500000])

        self.P_pv = np.array([0,0,0, 0, 0, 100, 150, 200, 300, 300, 300, 400, 400, 400, 400, 300, 200, 100 , 0, 0 , 0 ,0, 0, 0])
        
        self.P_grid = np.zeros(T)
    
        self.Price = np.array([0, 10, 10, 10 ,10 ,40, 40 ,40 ,40 ,30 ,20 ,10 ,10 ,30 ,40 ,10 ,20 ,10, 20 ,30 ,40 ,30,10 ,10])
        
        self.Qvalues ={}

    #Reward Part
    def get_reward(self,state, action,t):
        if action[0] == 1:
            self.P_discharge[t] = 0.0
        elif action[1] == 1:
            self.P_charge[t] = 0.0
        
        self.P_grid[t] = state[0] + self.P_charge[t] - self.P_discharge[t]
    
        #cost = price * max{0, P_grid[t]}
        return state[2] * max(0,self.P_grid[t])

    def get_Qvalue(self,state, action):
        return self.Qvalues.get((str(state),str(action)), 0.0)

    def get_baseline(self):
        return np.sum(np.multiply(self.Price, self.P_load))
    
    def update_Qvalue(self,pstate, action, reward, state):
        #Q-Learning
        #Qvalue = oldv + learning rate *{reward + discount rate * maxQ(s_t+1,a) - oldv}
        
        maxqnew = max([QLearning.get_Qvalue(self,state, a) for a in self.actions])
        value = reward + self.gamma * maxqnew
        
        oldv = self.Qvalues.get((str(pstate), str(action)), None)
        
        if oldv is None:
            self.Qvalues[(str(pstate),str(action))] = reward
        else:
            self.Qvalues[(str(pstate), str(action))] = oldv + self.alpha * (value - oldv)


    #Action Part
    def choose_action(self,state,t):
        #Epsilon-Greedy 
        if np.random.random() > self.epsilon:
            #Choose action randomly
            #return random.choice(self.actions)
            return self.actions[np.random.choice(len(self.actions))]
    
        else:
            q = [QLearning.get_Qvalue(self,state, a) for a in self.actions]
            maxQ = max(q)
            count = q.count(maxQ)
        
            if count > 1:
                best = [i for i in range(len(self.actions)) if q[i] == maxQ]
                #i = random.choice(best)
                i = best[np.random.choice(len(best))]
            else:
                i = q.index(maxQ)
                
        action = self.actions[i]
        return action



if __name__=="__main__":
    #Number of Learning
    learning_count = 20
    cost1 = []
    base_cost1 = []
    T = 24

    #TD Learning without Noise
    td = TD()
    S = [td.P_load - td.P_pv,td.E_st, td.Price]
    states = map(list,zip(*S))
    
    for i in range(learning_count):
        #Repeat for each time step t do
        s = 0
        for t in range(T-1):
            #Choose action for state in time t using "Epsilon-Greedy"
            action = td.choose_action(states[t],t)
        
            #Observe r[t+1], set reward
            reward = td.get_reward(states[t+1], action, t)
            s = s + reward
            
            #Take action a, observed reward r, and next state s'
            #delta <- reward + gamma * max{Q(s',a)} - Q(s,a)
            delta = reward + (td.gamma * max([td.get_Qvalue(str(states[t+1]), str(a)) for a in td.actions])) - td.get_Qvalue(str(states[t]), str(action))
            
            #Update for all state-action pair
        for j in range(T-1):
            for a in td.actions:
                td.update_Qvalue(states[t], action, reward, states[t+1],delta)
                td.update_Evalues(states[t], action)

        cost1.append(s)
        base_cost1.append(td.get_baseline())

    plt.subplot(221)
    plt.title("Temporal Difference Learning")
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.plot(cost1)
    plt.plot(base_cost1)    


    #TD Learning with Gaussian Noise
    td = TD()
    cost2 = []
    base_cost2 = []
    
    #Adding Noise by using Normal/Gaussian Distribution
    mu = np.mean(td.P_load)
    sigma = np.std(td.P_load)
    #noise = random.gauss(mu,sigma)
    noise = np.random.normal(mu, sigma, 24)
    
    td.P_load = td.P_load + noise
    S = [td.P_load - td.P_pv, td.E_st, td.Price]
    states = map(list,zip(*S))
        
    for i in range(learning_count):
        #Repeat for each time step t do
        s = 0
        for t in range(T-1):
            #Choose action for state in time t using "Epsilon-Greedy"
            action = td.choose_action(states[t],t)
        
            #Observe r[t+1], set reward
            reward = td.get_reward(states[t+1], action, t)
            s = s + reward
            
            #Take action a, observed reward r, and next state s'
            #delta <- reward + gamma * max{Q(s',a)} - Q(s,a)
            delta = reward + (td.gamma * max([td.get_Qvalue(str(states[t+1]), str(a)) for a in td.actions])) - td.get_Qvalue(str(states[t]), str(action))
            
            #Update for all state-action pair
        for j in range(T-1):
            for a in td.actions:
                td.update_Qvalue(states[t], action, reward, states[t+1],delta)
                td.update_Evalues(states[t], action)

        cost2.append(s)
        base_cost2.append(td.get_baseline())

    plt.subplot(222)
    plt.title("Temporal Difference Learning with Noise")
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.plot(cost2)
    plt.plot(base_cost2)  
    

    #QLearning without Noise
    cost3 = []
    base_cost3 = []
    ql = QLearning()
    #print "third,"
    #print ql.P_charge, ql.P_discharge
    
    S = [ql.P_load - ql.P_pv, ql.E_st, ql.Price]
    states = map(list,zip(*S))

    for i in range(learning_count):
        #Repeat for each time step t do
        s = 0
        for t in range(T-1):
            #Choose action for state in time t using "Epsilon-Greedy"
            action = ql.choose_action(states[t],t)
        
            #Observe r[t+1], set reward
            reward = ql.get_reward(states[t+1], action, t)
            s = s + reward
            
            #Take action a, observed reward r, and next state s'
            #Update for all state-action pair
        for j in range(T-1):
            for a in ql.actions:
                ql.update_Qvalue(states[t], action, reward, states[t+1])

        cost3.append(s)
        base_cost3.append(ql.get_baseline())

    plt.subplot(223)
    plt.title("Q Learning")
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.plot(cost3)
    plt.plot(base_cost3)



    #QLearning with Noise
    cost4 = []
    base_cost4 = []
    ql = QLearning()
    #print "fourth,"
    #print ql.P_charge, ql.P_discharge
    #Adding Noise by using Normal/Gaussian Distribution

    #noise = random.gauss(mu,sigma)
    mu = 0
    sigma = 0.5
    seed(200)
    noise = np.random.normal(mu, sigma,1)            
    for i in range(20):
        #Repeat for each time step t do
        s = 0

        #24hours
        for t in range(T-1):
            ql.P_load[t] = ql.P_load[t] + noise
            
            S = [ql.P_load - ql.P_pv, ql.E_st, ql.Price]
            states = map(list,zip(*S))
            
            #Choose action for state in time t using "Epsilon-Greedy"
            action = ql.choose_action(states[t],t)

            #Observe r[t+1], set reward
            reward = ql.get_reward(states[t+1], action, t)
            s = s + reward

            #Update
            ql.update_Qvalue(states[t], action, reward, states[t+1])

        cost4.append(s)
        base_cost4.append(ql.get_baseline())

    plt.subplot(224)
    plt.title("Q Learning with Noise")
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.plot(cost4)
    plt.plot(base_cost4)
    
    plt.show()
