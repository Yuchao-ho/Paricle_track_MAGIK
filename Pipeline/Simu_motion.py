import scipy.stats
import numpy as np
from stochastic.processes.noise import FractionalGaussianNoise as FGN

class models_phenom():
    def __init__(self):
        '''
        This class handles the generation of trajectories from different theoretical models. 
        ''' 
        # We define here the bounds of the anomalous exponent and diffusion coefficient
        self.bound_D = [1e-12, 1e6]
        self.bound_alpha = [0, 1.999]
        
        # We also define the value in which we consider directed motion
        self.alpha_directed = 1.9
        
        # Diffusion state labels: the position of each type defines its numerical label
        # i: immobile/trapped; c: confined; f: free-diffusive (normal and anomalous); d: directed
        self.lab_state = ['i', 'c', 'f', 'd']
    
    @staticmethod
    def gaussian(params:list|int, # If list, mu and sigma^2 of the gaussian. If int, we consider sigma = 0
                size = 1,  # Number of samples to get.
                bound = None # Bound of the Gaussian, if any.
                )-> np.array: # Samples from the given Gaussian distribution
        '''
        Samples from a Gaussian distribution of given parameters.
        '''
        # if we are given a single number, we consider equal to mean and variance = 0
        if isinstance(params, float) or isinstance(params, int):
            if size == 1:
                return params
            else:
                return np.array(params).repeat(size)
        else:
            mean, var = params        
            if bound is None:
                val = np.random.normal(mean, np.sqrt(var), size)
            if bound is not None:
                lower, upper = bound
                if var == 0: 
                    if mean > upper or mean < lower:
                        raise ValueError('Demanded value outside of range.')
                    val = np.ones(size)*mean
                else:
                    val = scipy.stats.truncnorm.rvs((lower-mean)/np.sqrt(var),
                                                    (upper-mean)/np.sqrt(var),
                                                    loc = mean,
                                                    scale = np.sqrt(var),
                                                    size = size)
            if size == 1:
                return val[0]
            else:
                return val

class models_phenom(models_phenom):

    @staticmethod
    def disp_fbm(alpha : float,
                 D : float,
                 T: int, 
                 deltaT : int = 1):
        
        # Generate displacements
        disp = FGN(hurst = alpha/2).sample(n = T)
        # Normalization factor
        disp *= np.sqrt(T)**(alpha)
        # Add D
        disp *= np.sqrt(2*D*deltaT)        
        
        return disp
    

    def _sample_diff_parameters(self, alphas : list, # List containing the parameters to sample anomalous exponent in state (adapt to sampling function)
                                Ds : list, # List containing the parameters to sample the diffusion coefficient in state (adapt to sampling function).
                                num_states : int, # Number of diffusive states.
                                epsilon_a : float, #  Minimum distance between anomalous exponents of various states.
                                gamma_d : float, # Factor between diffusion coefficient of various states.
                               ) : 

        alphas_traj = []
        Ds_traj = []
        for i in range(num_states): 

            # for the first state we just sample normally
            if i == 0:
                alphas_traj.append(float(self.gaussian(alphas[i], bound = models_phenom().bound_alpha)))
                Ds_traj.append(float(self.gaussian(Ds[i], bound = models_phenom().bound_D)))
           
            # For next states we take into account epsilon distance between diffusion
            # parameter
            else:
                ## Checking alpha
                alpha_state = float(self.gaussian(alphas[i], bound = models_phenom().bound_alpha))
                D_state = float(self.gaussian(Ds[i], bound = models_phenom().bound_D))

                if epsilon_a[i-1] != 0:
                    idx_while = 0
                    while models_phenom()._constraint_alpha(alphas_traj[-1], alpha_state, epsilon_a[i-1]):
                    #alphas_traj[-1] - alpha_state < epsilon_a[i-1]:
                        alpha_state = float(self.gaussian(alphas[i], bound = models_phenom().bound_alpha))                        
                        idx_while += 1
                        if idx_while > 100: # check that we are not stuck forever in the while loop
                            raise FileNotFoundError(f'Could not find correct alpha for state {i} in 100 steps. State distributions probably too close.')

                alphas_traj.append(alpha_state)
                
                ## Checking D
                if gamma_d[i-1] != 1:    
                    
                    idx_while = 0
                    while models_phenom()._constraint_d(Ds_traj[-1], D_state, gamma_d[i-1]):
                        D_state = float(self.gaussian(Ds[i], bound = models_phenom().bound_D))
                        idx_while += 1
                        if idx_while > 100: # check that we are not stuck forever in the while loop
                            raise FileNotFoundError(f'Could not find correct D for state {i} in 100 steps. State distributions probably too close.')
               
    
                Ds_traj.append(D_state)
                
        return alphas_traj, Ds_traj
    
    @staticmethod
    def _single_state_traj(T :int = 200, 
                          D : float = 1, 
                          alpha : float = 1, 
                          L : float = None,
                          deltaT : int = 1,
                          dim : int = 2):
        # Trajectory displacements
        disp_d = []
        for d in range(dim):
            disp_d.append(models_phenom().disp_fbm(alpha, D, T))
        # Labels
        lab_diff_state = np.ones(T)*models_phenom().lab_state.index('f') if alpha < models_phenom().alpha_directed else np.ones(T)*models_phenom().lab_state.index('d')
        labels = np.vstack((np.ones(T)*alpha, 
                            np.ones(T)*D,
                            lab_diff_state
                           )).transpose()

        # If there are no boundaries
        if not L:
            
            pos = np.vstack([np.cumsum(disp)-disp[0] for disp in disp_d]).transpose()
            
            return pos, labels

        # If there are, apply reflecting boundary conditions
        else:
            pos = np.zeros((T, dim))

            # Initialize the particle in a random position of the box
            pos[0, :] = np.random.rand(dim)*L
            for t in range(1, T):
                if dim == 2:
                    pos[t, :] = [pos[t-1, 0]+disp_d[0][t], 
                                 pos[t-1, 1]+disp_d[1][t]]            
                elif dim == 3:
                    pos[t, :] = [pos[t-1, 0]+disp_d[0][t], 
                                 pos[t-1, 1]+disp_d[1][t], 
                                 pos[t-1, 2]+disp_d[2][t]]            


                # Reflecting boundary conditions
                while np.max(pos[t, :])>L or np.min(pos[t, :])< 0: 
                    pos[t, pos[t, :] > L] = pos[t, pos[t, :] > L] - 2*(pos[t, pos[t, :] > L] - L)
                    pos[t, pos[t, :] < 0] = - pos[t, pos[t, :] < 0]

            return pos, labels

    
    def single_state(self,
                     N:int = 10,
                     T:int = 200, 
                     Ds:list = [1, 0], 
                     alphas:list = [1, 0], 
                     L:float = None,
                     dim:int = 2):

        positions = np.zeros((T, N, dim))
        labels = np.zeros((T, N, 3))

        for n in range(N):
            alpha_traj = self.gaussian(alphas, bound = self.bound_alpha)
            D_traj = self.gaussian(Ds, bound = self.bound_D)
            # Get trajectory from single traj function
            pos, lab = self._single_state_traj(T = T, 
                                               D = D_traj, 
                                               alpha = alpha_traj, 
                                               L = L,
                                               dim = dim
                                              )        
            positions[:, n, :] = pos
            labels[:, n, :] = lab

        return positions, labels
    
    @staticmethod
    def _multiple_state_traj(T = 200, 
                             M = [[0.95 , 0.05],[0.05 ,0.95]], 
                             Ds = [1, 0.1], 
                             alphas = [1, 1], 
                             L = None,
                             deltaT = 1,
                             return_state_num = False, 
                             init_state = None
                            ):
        
        # transform lists to numpy if needed
        if isinstance(M, list):
            M = np.array(M)
        if isinstance(Ds, list):
            Ds = np.array(Ds)
        if isinstance(alphas, list):
            alphas = np.array(alphas)


        pos = np.zeros((T, 2))
        if L: pos[0,:] = np.random.rand(2)*L

        # Diffusing state of the particle
        state = np.zeros(T).astype(int)
        if init_state is None:
            state[0] = np.random.randint(M.shape[0])
        else: state[0] = init_state
        
        # Init alphas, Ds
        alphas_t = np.array(alphas[state[0]]).repeat(T)
        Ds_t = np.array(Ds[state[0]]).repeat(T)
        
        
        # Trajectory displacements    
        dispx, dispy = [models_phenom().disp_fbm(alphas_t[0], Ds_t[0], T),
                        models_phenom().disp_fbm(alphas_t[0], Ds_t[0], T)]


        for t in range(1, T):

            pos[t, :] = [pos[t-1, 0]+dispx[t], pos[t-1, 1]+dispy[t]]  

            # at each time, check new state
            state[t] = np.random.choice(np.arange(M.shape[0]), p = M[state[t-1], :])


            if state[t] != state[t-1]:
                
                alphas_t[t:] =  np.array(alphas[state[t]]).repeat(T-t)  
                Ds_t[t:] = np.array(Ds[state[t]]).repeat(T-t)
                
                
                # Recalculate new displacements for next steps
                if len(dispx[t:]) > 1:                    
                    dispx[t:], dispy[t:] = [models_phenom().disp_fbm(alphas_t[t], Ds_t[t], T-t),
                                            models_phenom().disp_fbm(alphas_t[t], Ds_t[t], T-t)]
                        
                        
                else: 
                    dispx[t:], dispy[t:] = [np.sqrt(2*Ds[state[t]]*deltaT)*np.random.randn(), 
                                            np.sqrt(2*Ds[state[t]]*deltaT)*np.random.randn()]
                    

            if L is not None:
                # Reflecting boundary conditions
                while np.max(pos[t, :])>L or np.min(pos[t, :])< 0: 
                    pos[t, pos[t, :] > L] = pos[t, pos[t, :] > L] - 2*(pos[t, pos[t, :] > L] - L)
                    pos[t, pos[t, :] < 0] = - pos[t, pos[t, :] < 0]
                    
        # Define state of particles based on values of alphas: either free or directed
        label_diff_state = np.zeros_like(alphas_t)
        label_diff_state[alphas_t  < models_phenom().alpha_directed] = models_phenom().lab_state.index('f')
        label_diff_state[alphas_t >= models_phenom().alpha_directed] = models_phenom().lab_state.index('d')
                    
        if return_state_num:            
            return pos, np.array((alphas_t,
                                  Ds_t,
                                  label_diff_state,
                                  state)).transpose()
        else: 
            return pos, np.array((alphas_t,
                                  Ds_t,
                                  label_diff_state)).transpose()

    def multi_state(self,
                    N = 10,
                    T = 200,
                    M: np.array = [[0.9 , 0.1],[0.1 ,0.9]],
                    Ds: np.array = [[1, 0], [0.1, 0]], 
                    alphas: np.array = [[1, 0], [1, 0]], 
                    gamma_d = None, 
                    epsilon_a = None, 
                    L = None,
                    return_state_num = False,
                    init_state = None): 
        
        # transform lists to numpy if needed
        if isinstance(M, list):
            M = np.array(M)
        if isinstance(Ds, list):
            Ds = np.array(Ds)
        if isinstance(alphas, list):
            alphas = np.array(alphas)
        
        
        # Get epsilon and gamma
        if gamma_d is None:
            gamma_d = [1]*(M.shape[0]-1)
        if epsilon_a is None:
            epsilon_a = [0]*(M.shape[0]-1)
        

        trajs = np.zeros((T, N, 2))
        if return_state_num:
            labels = np.zeros((T, N, 4))
        else:
            labels = np.zeros((T, N, 3))

        for n in range(N):
            
            ### Sampling diffusion parameters for each state
            alphas_traj = []
            Ds_traj = []
            
            alphas_traj, Ds_traj = self._sample_diff_parameters(alphas = alphas,
                                                                Ds = Ds,
                                                                num_states = M.shape[0],
                                                                epsilon_a = epsilon_a,
                                                                gamma_d = gamma_d)
                    
            #### Get trajectory from single traj function
            traj, lab = self._multiple_state_traj(T = T,
                                                  L = L,
                                                  M = M,
                                                  alphas = alphas_traj,
                                                  Ds = Ds_traj,
                                                  return_state_num = return_state_num,
                                                  init_state = init_state
                                                 )  
                
            trajs[:, n, :] = traj
            labels[:, n, :] = lab 
            
        return trajs, labels


    