import ActorCritic
import environment
import numpy as np
import theano
import theano.tensor as T

def getE(iteration):
    if iteration<20:
        return 0.8
    elif iteration<80:
        return 0.5
    elif iteration<160:
        return 0.3
    else:
        return 0.2

def goodactions():
    a = np.zeros((10,12))
    b = np.array(range(12))
    np.random.shuffle(b)
    for i in range(12):
        if i<9:
            a[i,b[i]]=1
        else:
            a[9,b[i]]=1
    return a
    
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
        elif np.random.rand()>0.5:
            action.append(goodactions())
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
    getdata(iteration,env,ActionReplay,Learner,n=500)
    for i in range(2):
        X,_,_,_ = ActionReplay.get(n=512)
        print Learner.ActorTrain(X,BS=8)
    for i in range(10):
        X,Target,Action,Reward = ActionReplay.get(n=512)
        print Learner.CriticTrain(X,Action,Reward,TLearner.CriticStateValue(Target,Learner.Action(Target)),BS=16)     #Implementing Double DQN
        print Learner.CriticStateValue(X,Action).mean(),Reward.mean(),TLearner.CriticStateValue(Target,Learner.Action(Target)).mean()
    
    
def main():
    env = environment.Environment()
    ActionReplay = environment.ActionReplay()
    Learner = ActorCritic.ReinforcementLearner()
    TLearner = ActorCritic.ReinforcementLearner()
    for i in range(400):
        Train(i,env,ActionReplay,Learner,TLearner)
    
if __name__ == '__main__':
    main()