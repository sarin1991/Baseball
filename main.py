import ActorCritic
import environment
import numpy as np
import theano
import theano.tensor as T

def getE(iteration):
    if iteration<10:
        return 1
    elif iteration<50:
        return 0.5
    elif iteration<120:
        return 0.2
    elif iteration<160:
        return 0.1
    else:
        return 0.01

def getdata(iteration,env,ActionReplay,Learner,n=100):
    target = []
    x = []
    reward = []
    action = []
    for i in range(n):
        x.append(env.state)
        E = getE(iteration)
        if np.random.rand()>E:
            action.append(Learner.Action(env.state))
        else:
            action.append(np.random.randint(2,size=(10,12)))
        reward.append(env.updatestate(action[-1]))
        target.append(env.state)
    Target = np.asarray(target)
    X = np.asarray(x)
    Reward = np.asarray(reward)
    Action = np.asarray(action)
    del target,x,reward,action
    ActionReplay.put(X=X,Target=Target,Action=Action,Reward=Reward)
    del X,Target,Reward,Action

def Train(iteration,env,ActionReplay,Learner,TLearner):
    TLearner.setstate(Learner.getstate())
    getdata(iteration,env,ActionReplay,Learner)
    for i in range(10):
        X,Target,Action,Reward = ActionReplay.get()
        #print X.shape,Target.shape,Action.shape,Reward.shape 
        print Learner.Train(X,Action,Reward,TLearner.CriticStateValue(Target,Learner.Action(Target)),BS=1000)     #Implementing Double DQN
    #for i in range(5):
    #    X,_,_,_ = ActionReplay.get()
    #    print Learner.ActorTrain(X)
    
def main():
    env = environment.Environment()
    ActionReplay = environment.ActionReplay()
    Learner = ActorCritic.ReinforcementLearner()
    TLearner = ActorCritic.ReinforcementLearner()
    for i in range(200):
        Train(i,env,ActionReplay,Learner,TLearner)
    
if __name__ == '__main__':
    main()