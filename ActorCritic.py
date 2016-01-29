import lasagne
import theano
import theano.tensor as T
import numpy as np
import BatchNormalization as BN

class Actor():
    '''Given State Predict Actor
       State: NoneX100X10X12
       Actor: NoneX10X12'''
    def __init__(self,StateVar):
        self.LearningRate = 0.01
        self.InputVar = StateVar
        self.network = self.BuildNetwork()
        self.Predict = self.getPredict()
        self.OutputVar = lasagne.layers.get_output(self.network)
        
    def BuildNetwork(self):
        network = lasagne.layers.InputLayer(shape=(None,10,10,12),input_var=self.InputVar,name='0')
        network = lasagne.layers.DenseLayer(network,num_units=1500,W=lasagne.init.GlorotUniform(gain='relu'),nonlinearity=lasagne.nonlinearities.leaky_rectify,name = '1')
        network = BN.batch_norm(network,name = '1')
        #network = lasagne.layers.DropoutLayer(network,0.8)
        
        network = lasagne.layers.DenseLayer(network,num_units=1000,W=lasagne.init.GlorotUniform(gain='relu'),nonlinearity=lasagne.nonlinearities.leaky_rectify,name = '2')
        network = BN.batch_norm(network,name = '2')
        #network = lasagne.layers.DropoutLayer(network)
        
        network = lasagne.layers.DenseLayer(network,num_units=500,W=lasagne.init.GlorotUniform(gain='relu'),nonlinearity=lasagne.nonlinearities.leaky_rectify,name = '3')
        network = BN.batch_norm(network,name = '3')
        #network = lasagne.layers.DropoutLayer(network)
        
        network = lasagne.layers.DenseLayer(network,num_units=300,W=lasagne.init.GlorotUniform(gain='relu'),nonlinearity=lasagne.nonlinearities.leaky_rectify,name = '4')
        network = BN.batch_norm(network,name = '4')
        #network = lasagne.layers.DropoutLayer(network)
        
        network = lasagne.layers.DenseLayer(network,num_units=120,W=lasagne.init.GlorotUniform(gain=1.0),nonlinearity=lasagne.nonlinearities.sigmoid,name = '5')
        network = BN.batch_norm(network,name = '5')
        
        network = lasagne.layers.ReshapeLayer(network,shape=([0],10,12),name='5')
        return network
        
    def getPredict(self):
        output = lasagne.layers.get_output(self.network,deterministic=True)
        prediction = T.switch(output<0.5,0,1)
        predict = theano.function([self.InputVar],prediction,allow_input_downcast=True)
        return predict
        
    def getstate(self):
        return self.LearningRate,lasagne.layers.get_all_param_values(self.network)
        
    def setstate(self,state):
        self.LearningRate,NetworkState = state
        lasagne.layers.set_all_param_values(self.network,NetworkState)
        
class Critic():
    '''Predict Q-Value given Action and State'''
    def __init__(self,ActVar,StateVar):
        self.LearningRate = 100
        self.gamma = 0.8
        self.ActVar = ActVar
        self.StateVar = StateVar
        self.RewardVar = T.vector('Reward')
        self.TargetQValueVar = T.matrix('TargetQValueVar')
        self.network = self.BuildNetwork()
        self._Train = self.getTrain()
        self.StateValue = self.getStateValue()
        
    def getstate(self):
        return self.LearningRate,self.gamma,lasagne.layers.get_all_param_values(self.network)
        
    def setstate(self,state):
        self.LearningRate,self.gamma,NetworkState = state
        lasagne.layers.set_all_param_values(self.network,NetworkState)
    
    def Train(self,X,Action,Reward,Target,BS = 128):
            error = []
            N = len(X)/BS
            for i in range(N):
                error.append(self._Train(X[i*BS:(i+1)*BS],Action[i*BS:(i+1)*BS],Reward[i*BS:(i+1)*BS],Target[i*BS:(i+1)*BS]))
            if N!=0 and len(X[(i+1)*BS:])!=0:
                error.append(self._Train(X[(i+1)*BS:],Action[(i+1)*BS:],Reward[(i+1)*BS:],Target[(i+1)*BS:]))
            elif N==0:
                error.append(self._Train(X,Action,Reward,Target))
            return np.mean(error)
                
    def BuildNetwork(self):
        ActVar = T.reshape(self.ActVar,(-1,1,10,12))
        ActInputLayer = lasagne.layers.InputLayer(shape=(None,1,10,12),input_var=ActVar,name='0')
        StateInputLayer = lasagne.layers.InputLayer(shape=(None,10,10,12),input_var=self.StateVar,name='0')
        
        network = lasagne.layers.ConcatLayer([ActInputLayer,StateInputLayer],axis=1)
        network = lasagne.layers.DenseLayer(network,num_units=2000,W=lasagne.init.GlorotUniform(gain='relu'),nonlinearity=lasagne.nonlinearities.leaky_rectify,name = '1')        
        network = BN.batch_norm(network,name = '1')
        #network = lasagne.layers.DropoutLayer(network)
        
        network = lasagne.layers.DenseLayer(network,num_units=1000,W=lasagne.init.GlorotUniform(gain='relu'),nonlinearity=lasagne.nonlinearities.leaky_rectify,name = '2')
        network = BN.batch_norm(network,name = '2')
        #network = lasagne.layers.DropoutLayer(network)
        
        network = lasagne.layers.DenseLayer(network,num_units=500,W=lasagne.init.GlorotUniform(gain='relu'),nonlinearity=lasagne.nonlinearities.leaky_rectify,name = '3')
        network = BN.batch_norm(network,name = '3')
        #network = lasagne.layers.DropoutLayer(network)
        
        network = lasagne.layers.DenseLayer(network,num_units=40,W=lasagne.init.GlorotUniform(gain='relu'),nonlinearity=lasagne.nonlinearities.leaky_rectify,name = '4')
        network = BN.batch_norm(network,name = '4')
        #network = lasagne.layers.DropoutLayer(network)
        
        network = lasagne.layers.DenseLayer(network,num_units=5,W=lasagne.init.GlorotUniform(gain='relu'),nonlinearity=lasagne.nonlinearities.leaky_rectify,name = '5')
        network = BN.batch_norm(network,name = '5')
        #network = lasagne.layers.DropoutLayer(network)
        
        network = lasagne.layers.DenseLayer(network,num_units=1,W=lasagne.init.GlorotUniform(gain='relu'),nonlinearity=lasagne.nonlinearities.leaky_rectify,name = '6')
        network = BN.batch_norm(network,name = '6')
        
        return network
        
    def getTrain(self):
        Prediction = lasagne.layers.get_output(self.network)
        TargetQValue = self.RewardVar + self.gamma*self.TargetQValueVar.T
        loss = lasagne.objectives.squared_error(TargetQValue.T,Prediction).mean()
        params = lasagne.layers.get_all_params(self.network,trainable=True)
        updates = lasagne.updates.adam(loss,params,self.LearningRate)
        Train = theano.function([self.StateVar,self.ActVar,self.RewardVar,self.TargetQValueVar],loss,updates=updates,allow_input_downcast=True)
        return Train
        
    def getStateValue(self):
        Prediction = lasagne.layers.get_output(self.network,deterministic=True)
        StateValue = theano.function([self.StateVar,self.ActVar],Prediction,allow_input_downcast=True)
        return StateValue
        
class ReinforcementLearner():
    def __init__(self):
        self.StateVar = T.tensor4('StateVar')
        self.actor = Actor(self.StateVar)
        self.ActVar = self.actor.OutputVar
        self.critic = Critic(ActVar = self.ActVar,StateVar = self.StateVar)
        self.CriticTrain = self.critic.Train
        self._ActorTrain = self.getActorTrain()
        self.CriticStateValue = self.critic.StateValue
    
    def getstate(self):
        return self.actor.getstate(),self.critic.getstate()
        
    def setstate(self,state):
        ActorState,CriticState = state
        self.actor.setstate(ActorState)
        self.critic.setstate(CriticState)
    
    def Train(self,X,Action,Reward,Target,BS = 128):
        Aerror = []
        Cerror = []
        N = len(X)/BS
        for i in range(N):
            Aerror.append(self.ActorTrain(X[i*BS:(i+1)*BS],BS=BS))
            Cerror.append(self.CriticTrain(X[i*BS:(i+1)*BS],Action[i*BS:(i+1)*BS],Reward[i*BS:(i+1)*BS],Target[i*BS:(i+1)*BS]))
        if N!=0 and len(X[(i+1)*BS:])!=0:
            Aerror.append(self.ActorTrain(X[(i+1)*BS:],BS=BS))
            Cerror.append(self.CriticTrain(X[(i+1)*BS:],Action[(i+1)*BS:],Reward[(i+1)*BS:],Target[(i+1)*BS:],BS=BS))
        elif N==0:
            Aerror.append(self.ActorTrain(X,BS=BS))
            Cerror.append(self.CriticTrain(X,Action,Reward,Target,BS=BS))
        return np.mean(Aerror),np.mean(Cerror)
        
    def ActorTrain(self,X,BS=128):
        error = []
        N = len(X)/BS
        for i in range(N):
            error.append(self._ActorTrain(X[i*BS:(i+1)*BS]))
        if N!=0 and len(X[(i+1)*BS:])!=0:
            error.append(self._ActorTrain(X[(i+1)*BS:]))
        elif N==0:
            error.append(self._ActorTrain(X))
        return np.mean(error)
                    
    def getActorTrain(self):
        CriticOutput = lasagne.layers.get_output(self.critic.network)
        loss = -1*CriticOutput
        loss = loss.mean()
        params = lasagne.layers.get_all_params(self.actor.network,trainable=True)
        updates = lasagne.updates.adam(loss,params,self.actor.LearningRate)
        Train = theano.function([self.StateVar],loss,updates=updates,allow_input_downcast=True)
        return Train
        
    def Action(self,State):
        if State.shape == (10,10,12):
            State = np.expand_dims(State,axis=0)
            return self.actor.Predict(State)[0]
        return self.actor.Predict(State)