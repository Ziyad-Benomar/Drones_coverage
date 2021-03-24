# INF581 Final project #

## Run the environment
To do a simulation of the RL, all you need to do is to run the file "run.py", with the command
Make sure you have all the needed packages installed
- AiGym
- Pygame
- Numpy
- Pytorch

In our code, we did not manage the eventuality that some Pytorch tensors might be stored in the GPU. Hence, you might find some issues running our code if you have a GPU.
To overcome this, run the command (on a Unix like environment)  
   export CUDA_VISIBLE_DEVICES=""  
to ensure that all computations will be done on the CPU.  


## COVERING

This environment is represented by a matrix (zone) of dimensions (n,m) and N drones moving on its cells.  

Each cell in the zone is either free, either containing an obstacle, either containing a target (a human)  
The objective of the drones is to cover the zone as fast as possible and find all the targets.  
drones can walk in cells containing targets, but not in cells containing obstacles  

For each i in range(n) and j in range(m), we have (i being in the ordinate axis, and j in the abscissa axis)  
- zone[i,j] = 0 iif the cell free  
- zone[i,j] = -1 iif the cell contains an obstacle  
- zone[i,j] = 1 iif the cell contains a target  

Conventions :  
- x coordinates increase from left to right, y coordinates increase from top to bottom  

Notes :
- There are still some small details to be handled in the code, we can discuss them later  
- I didn't do the part concerning connection and exchanging data between the drones... I defined some attributes and I wrote some functions for that, but they must be reviewed   


## How to use this environment ?  
- Make sure to have gym and pygame installed  
- In the main folder, run the command "pip install -e ."  
- ET VOILÀ !  
- run the file "run.py" to experiment the environment  

## Understand the vizualisation :  
- dark cells are the obstacles  
- green cells are the targets  
- black circles are the drones (moving in the zone)  
- the blue rectangles around them are their action spaces  
- the red rectangles around them are their vision fields  
- cells with red crosses on them are the ones that have NOT been covered yet  
- cells with purple stars on them have just been covered after the last move (they are the new observation)  

=======
# MARL  
Multi-Agent Reinforcment Learning   


The idea of this git project is to compute the MARL algorithm releated to drone cooperation.   

STEP :  

- Define the problem with the vocabulary of MARL :  
    1) Go in the environment and check the information that can be used for the algorithm.   
    2) Define the value function. In a first time we can work with a value function that does not integrate any detection reward.  
        . Define the value function : We first need to translate function f as the good one considering the fact that drones look in front of them.   

- Creation of the MARL process :   
    1) Create the integration pipeline (Data extraction / Data processing).   
    2) Create the MARL pipeline.  

    

What information do we need for MARL algorithm :   
    
-   Constants :  

        . learning_rate
        . schedule 
        . discount_factor

-   UAV's :   
        . s = position 
        . parameters theta
        . Space of actions A_k

We need to be able to compute the part of the overlapping area.   
   


Pseudo code for the main algorithm : 

- Input : learning_rate, discount_factor, schedule, n_step, vector_function  
---------------------------------------------------------------------------  

Initialize the vector theta_{i,0}   

for episode in range(n_episode):  

        Randomly initialize  state (position) s_{i,0}  

        for k in range(n_time per episode):  

                for i in range(n_vehicule):  

                        Gather state information (position matrix)  
                        Take U uniform and if U < e_k (schedule):   
                                Choose random joint action  
                        Else :                                                                                                                  
                                Find an optimal joint action* -----------------------   
                                                                                    |  
                                                                                    |  
                                                                                    |       
                                                                                    |  
                 In order to do so we muste use the social convention rule*         |  
                                                                                    |  
                 We need a solver      <---------------------------------------------  


                        Update position matrix  
                        Compute global reward  
                        Update theta_{i,k+1}    

Output : theta vector updated and policy pi.   



Social convention rule :   
-----------------------  
We specify a ranking order between each UAV. The one with higher order can choose first his action and can modify possible action for other (priority rule). It means that we must update the matrix A_k at each step.   



Find an optimal joint action :   
------------------------------  
We want to find the best action possible according to the overall reward. According to the article the idea is to solve a linear problem.  
