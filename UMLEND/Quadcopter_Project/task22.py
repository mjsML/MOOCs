import numpy as np
from physics_sim import PhysicsSim
from scipy.spatial import distance
class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3
    
        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4
        #Penalize not moving along the Z axis .
        #self.beta=[1,1,.1]

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        ########
        #Approach 1
        #######
        return np.tanh(1 - 0.0005*abs(self.sim.pose[:3] - self.target_pos).sum())
    
        ########
        #Approach 2
        #######
        #reward = 0
        #penalty = 0
        #current_pos = self.sim.pose[:3]
        # penalty for euler angles, we want the takeoff to be stable
        #penalty += abs(self.sim.pose[3:6]).sum()
        
        # penalty for distance squared from target * param beta (self.beta)
        #penalty += self.beta[0]*distance.euclidean(current_pos[0],self.target_pos[0])**2
        #penalty += self.beta[1]*distance.euclidean(current_pos[1],self.target_pos[1])**2
        #penalty += self.beta[2]*distance.euclidean(current_pos[2],self.target_pos[2])**2
        
        #https://en.wikipedia.org/wiki/Euclidean_distance
        #https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.euclidean.html
        #EuclideanDistance=distance.euclidean(current_pos,self.target_pos)
        
        # Stronger signal if near target
        #if EuclideanDistance < 1:
            #reward += 100
        # constant reward for flying
        #reward += 1
        #return reward - penalty
        
        ########
        #Approach 3
        #######
        #return np.tanh(1 - 0.1-numpy.linalg.norm(self.sim.pose[:3] , self.target_pos))
        
        ########
        #Approach 4
        #######
        #return np.tanh(1 - 0.05-numpy.linalg.norm(self.sim.pose[:3] , self.target_pos))

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state