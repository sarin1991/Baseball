import numpy as np

class Environment():
    '''
    Action Rows:
    First 1 is pitcher
    Next 1 is catcher
    Rest 7 are 1b, 2b, 3b, CF, LF, Rf, SS
    Last 1 is left out
    
    Action Columns:
    First 3 are both pitchers and catchers
    Next 4 are just pitchers
    Next 1 is just a catcher
    Then the rest 4
    '''
    def __init__(self):
        self.state = np.zeros((100,10,12))
        self.pitcher = [0,1,2,3,4,5,6]
        self.catcher = [0,1,2,7]
        self.balancer = np.zeros((10,12))
        
    def getstate(self):
        return self.state
        
    def setstate(self,state):
        self.state = state
    
    def _getcost(self,action):
        '''action:10X12'''
        C = 0
        if not np.all(np.sum(action[:9],axis=1)==1):
            C = C - 1000
        if not np.all(np.sum(action,axis=0)==1):
            C = C - 1000
        if not np.sum(action[0,self.pitcher])==1:
            C = C - 10
        if not np.sum(action[1,self.catcher])==1:
            C = C - 10
        if np.any((action[-1]+self.state[0,-1])>1):
            C = C - 1
        C = C - 0.2*(self.balancer*action).sum()
        self.balancer = action+0.8*self.balancer
        return C
        
    def updatestate(self,action):
        '''action:10X12,state:100X10X12'''
        newstate = np.concatenate([np.expand_dims(action,axis=0),self.state[:99]],axis=0)
        cost = self._getcost(action)
        self.state = newstate
        return cost

class ActionReplay():
    def __init__(self):
        self.X = np.zeros((5000,100,10,12))
        self.Target = np.zeros(self.X.shape)
        self.Action = np.zeros((len(self.X),10,12))
        self.Reward = np.zeros(len(self.X))
        self.N = 0
        self.Index = np.array(range(len(self.X)))
        
    def get(self,n=1000):
        if self.N<n:
            return self.X[:self.N],self.Target[:self.N],self.Action[:self.N],self.Reward[:self.N]
        elif self.N<len(self.Reward):
            Index = self.Index[:self.N]
            np.random.shuffle(Index)
            return self.X[Index[:n]],self.Target[Index[:n]],self.Action[Index[:n]],self.Reward[Index[:n]]
        else:
            np.random.shuffle(self.Index)
            return self.X[self.Index[:n]],self.Target[self.Index[:n]],self.Action[self.Index[:n]],self.Reward[self.Index[:n]]
    
    def put(self,X,Target,Action,Reward):
        if not len(X)==len(Target)==len(Reward):
            raise ValueError('Input Shapes do not match')
        if self.N<len(self.Index):
            if self.N+len(X)<=len(self.Index):
                Index = self.Index[self.N:(self.N+len(X))]
                self.X[Index] = X
                self.Target[Index] = Target
                self.Action[Index] = Action
                self.Reward[Index] = Reward
                self.N = self.N + len(X)
            else:
                N = len(self.X) - self.N
                self.X[self.N:] = X[:N]
                self.Target[self.N:] = Target[:N]
                self.Action[self.N:] = Action[:N]
                self.Reward[self.N:] = Reward[:N]
                self._shuffleput(X[N:],Target[N:],Action[N:],Reward[N:])
        else:
            self._shuffleput(X,Target,Action,Reward)
    
    def _shuffleput(self,X,Target,Action,Reward):
        np.random.shuffle(self.Index)
        Index = self.Index[:len(X)]
        self.X[Index] = X
        self.Target[Index] = Target
        self.Action[Index] = Action
        self.Reward[Index] = Reward
        
        