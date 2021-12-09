# IcarusGym

IcarusGym is an agent-oriented information-centric (ICN) network caching simulator based on 
[Icarus](https://icarus-sim.github.io) and [OpenAI Gym](https://gym.openai.com/). 

It is designed for users who want to apply reinforcement learning (RL) in researches on ICN caching. 

IcarusGym exploits **[GymProxy](https://github.com/etri/GymProxy)** to make Icarus and Gym inter-operate with each 
other. 

It also exploits **[python-priorityq](https://github.com/elplatt/python-priorityq)** for implementation 
convenience (Included in icarusgym.from_contribs package).

We recommend you to understand Icarus and OpenAI Gym before using IcarusGym. 

## Installation

As pre-requisite, you should have Python 3.7+ installed on your machine. 

[Icarus](https://icarus-sim.github.io), [OpenAI Gym](https://gym.openai.com/), and 
[GymProxy](https://github.com/etri/GymProxy) libraries are also required. 

Clone this repository on your machine and run: 
    
    $ cd ~/projects/icarusgym   # We assume that the repository is cloned to this directory
    $ pip install . 

If you use Anaconda, you can install IcarusGym by the followings:

    $ conda activate my_env     # We assume that 'my_env' is your working environment
    $ conda develop ~/projects/icarusgym

## Usage Examples

We present three gym-type environments as usage examples of IcarusGym: 
- DecisionArrayCache
    - Implemented based on the following reference: 
        - A. Sadeghi et al., "Deep reinforcement learning for adaptive caching in hierarchical content delivery 
        networks," IEEE Trans. Cogn. Commun. Netw., vol. 5, no. 4, pp. 1024-1033, Dec. 2019.
- PassiveAgentCache
    - Just observes the actions of a legacy caching algorithms provided by the original Icarus. It is useful when you 
    need to compare the performances of reinforcement learning-based control and legacy control in same scenario.
- TtlCache
    - Implemented based on the following reference: 
        - M. Dehghan et al., "A utility optimization approach to network cache design," IEEE/ACM Trans. Netw., vol. 27, 
        no. 3, pp. 1013-1027, May 2019 (Earlier version is presented in IEEE INFOCOM 2016).

## Acknowledgement

This work was supported by the Institute of Information and Communications Technology Planning and Evaluation (IITP)
and funded by the Korea government (MSIT) under Grant No. 2017-0-00045, Hyper-Connected Intelligent Infrastructure
Technology Development. 
