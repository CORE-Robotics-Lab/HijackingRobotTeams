import numpy as np
from multiagent.core_ff import World, Agent, Landmark, Wall
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 16 # original 4
        # world.contact_force = 1e+100
        num_good_agents = 1
        num_adversaries = 3
        num_PO_adv = 2
        num_agents = num_adversaries + num_good_agents + num_PO_adv
        num_landmarks = 3
        num_walls = 4
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            # set agent type: adversary, prey and capture
            if i < num_adversaries:
                agent.adversary = True
                agent.prey = False
                agent.PO_adv = False
                agent.capture = False
                # INFO: Let's assume predators can give communication
                agent.silent = False
            elif i < (num_adversaries + num_good_agents):
                agent.adversary = False
                agent.prey = True
                agent.PO_adv = False
                agent.capture = False
                agent.action_callback = random_walk
            else:
                agent.adversary = False
                agent.prey = False
                agent.PO_adv = True
                agent.capture = False
            # agent.adversary = True if i < num_adversaries else False
            # set dynamic constraints
            if agent.adversary:
                agent.size = 0.075
                agent.accel = 3.0
                agent.max_speed = 1.0
            elif agent.prey:
                agent.size = 0.05
                agent.accel = 4
                agent.max_speed = 1.3
            else:
                agent.size = 0.1
                agent.accel = 3.0
                agent.max_speed = 1.0
            # agent.size = 0.075 if agent.adversary else 0.05
            # agent.accel = 3.0 if agent.adversary else 4.0
            # #agent.accel = 20.0 if agent.adversary else 25.0
            # agent.max_speed = 1.0 if agent.adversary else 1.2
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.2
            landmark.boundary = False

        wall_pos = 1.0
        self.wall_pos = wall_pos
        world.walls.append(Wall(orient='H', axis_pos=-wall_pos, endpoints=(-wall_pos, wall_pos), width=0.01, hard=True))
        world.walls.append(Wall(orient='H', axis_pos=wall_pos, endpoints=(-wall_pos, wall_pos), width=0.01, hard=True))
        world.walls.append(Wall(orient='V', axis_pos=-wall_pos, endpoints=(-wall_pos, wall_pos), width=0.01, hard=True))
        world.walls.append(Wall(orient='V', axis_pos=wall_pos, endpoints=(-wall_pos, wall_pos), width=0.01, hard=True))
        for i, wall in enumerate(world.walls):
            wall.name = 'wall %d' % i
            wall.collide = True
            wall.movable = False
            wall.size = 0.01
            wall.boundary = True
        # make initial conditions


# class Wall(object):
#     def __init__(self, orient='H', axis_pos=0.0, endpoints=(-1, 1), width=0.1, hard=True):
#         # orientation: 'H'orizontal or 'V'ertical
#         self.orient = orient
#         # position along axis which wall lays on (y-axis for H, x-axis for V)
#         self.axis_pos = axis_pos
#         # endpoints of wall (x-coords for H, y-coords for V)
#         self.endpoints = np.array(endpoints)
#         # width of wall
#         self.width = width
#         # whether wall is impassable to all agents
#         self.hard = hard
#         # color of wall
#         self.color = np.array([0.0, 0.0, 0.0])



        self.reset_world(world)
        return world


    def reset_world(self, world, env_seed = None):
        # random properties for agents
        if env_seed is not None:
            np.random.seed(env_seed)
        for i, agent in enumerate(world.agents):
            if agent.adversary:
                agent.color = np.array([0.85, 0.35, 0.35])
            elif agent.prey:
                agent.color = np.array([0.35, 0.85, 0.35])
            # elif i == 4:
            #     agent.color = np.array([0.35, 0.35, 0.85])
            else:
                agent.color = np.array([1, 1, 0])
            # agent.color = np.array([0.35, 0.85, 0.35]) if not agent.adversary else np.array([0.85, 0.35, 0.35])
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = np.random.uniform(-0.8*self.wall_pos, +0.8*self.wall_pos, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)


    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        elif agent.PO_adv:
            collisions = 0
            capture_num = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
                    capture_num += 1
            return [collisions, capture_num]  
        elif agent.prey:
            collisions=0
            capture_num=0
            for adv in self.adversaries(world):
                if self.is_collision(agent, adv):
                    collisions += 1
            for cap in self.po_predator(world):
                if self.is_collision(cap, agent):
                    # collisions += 1
                    capture_num += 1
            return [collisions, capture_num]                
        else:
            return 0


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def is_capture(self, adv, ag):
        if self.is_collision(adv, ag) and (adv.action.cp[0] == 1):
            return True
        else:
            return False

    def capturing(self, cap):
        return cap.action.cp[0] == 1
             

    # return all preys
    def good_agents(self, world):
        return [agent for agent in world.agents if agent.prey]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    # return all capture agents
    def po_predator(self, world):
        return [agent for agent in world.agents if agent.PO_adv]

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        if agent.adversary:
            main_reward = self.adversary_reward(agent, world)
        elif agent.prey:
            main_reward = self.agent_reward(agent, world)
        else:
            main_reward = self.capture_reward(agent, world)
        # main_reward = self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)
        return main_reward

    def agent_reward(self, agent, world):
        # Agents are negatively rewarded if caught by capture agents
        rew = 0
        shape = True
        adversaries = self.adversaries(world)
        captures = self.po_predator(world)
        adver_cap = adversaries + captures
        if shape:  # reward can optionally be shaped (increased reward for increased distance from adversary)
            for adv in adver_cap:
                rew += 0.1 * np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos))) # original 0.1
        if agent.collide:
            for a in adversaries:
                if self.is_collision(a, agent):
                    rew -= 0.1
            for a in captures:
                if self.is_collision(a, agent):
                    rew -= 0.1
                

        # agents are penalized for exiting the screen, so that they can be caught by the adversaries
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 100
            # return min(10 * np.exp(3 * x - 3), 500)
            return 0
        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])/self.wall_pos
            rew -= bound(x)

        return rew

        # def bound(x):
        #     if x < 0.9:
        #         return 0
        #     if x < 1.0:
        #         return (x - 0.9) * 10
        #     return min(np.exp(2 * x - 2), 10)
        # for p in range(world.dim_p):
        #     x = abs(agent.state.p_pos[p])
        #     rew -= bound(x)

        # return rew

    def adversary_reward(self, agent, world):
        # Adversaries are rewarded for collisions with agents
        rew = 0
        shape = True
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        captures = self.po_predator(world)
        if shape:  # reward can optionally be shaped (decreased reward for increased distance from agents)
            # for adv in adversaries:
            rew -= 0.7 * min([np.sqrt(np.sum(np.square(a.state.p_pos - agent.state.p_pos))) for a in agents]) # original -0.4
        if agent.collide:
            for ag in agents:
                # for adv in adversaries:
                if self.is_collision(ag, agent):
                    rew += 0.5 # original: 10
                for cap in captures:
                    if self.is_collision(ag, cap):
                        rew += 0.5 # original: 1
                # else:
                #     rew -= 0.05

        return rew

    def capture_reward(self, agent, world):
        rew = 0
        shape = True
        agents = self.good_agents(world)
        if shape:  # reward can optionally be shaped (decreased reward for increased distance from agents)
            rew -= 0.7 * min([np.sqrt(np.sum(np.square(a.state.p_pos - agent.state.p_pos))) for a in agents]) # original coefficient -0.0
        if agent.collide:
            for ag in agents:
                if self.is_collision(ag, agent):
                    rew += 0.5 
        return rew       

    # INFO: Capturer cannot see anything
    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent: continue
            if agent.PO_adv and other.adversary:
                comm.append(other.state.c) # include the other agents' states as communications, but not sure if it is an zero array
            # if other.prey:
            if not agent.PO_adv:
                other_pos.append(other.state.p_pos - agent.state.p_pos) # observe the distance to other agents
                # other_vel.append(other.state.p_vel)# predators and capturers can observe the velocity of prey and not vice versa / only prey's velocity is observable
            else:
                other_pos.append(np.zeros_like(other.state.p_pos - agent.state.p_pos))
                # other_vel.append(np.zeros_like(other.state.p_vel))
            if other.prey:
                if not agent.PO_adv:
                    other_vel.append(other.state.p_vel)
                else:
                    other_vel.append(np.zeros_like(other.state.p_vel))


        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel + comm) # adversary return dim (16/14=2+2+4+6+2/0(prey),)


def random_walk(agent, world):

    # agent_action = Action()
    wall_pos = np.abs(world.walls[0].axis_pos)
    acc_limit = 5.0
    time_step_limit = 20
    # raw_acc = np.random.randn(*agent.action.u.shape)
    raw_acc = np.zeros(world.dim_p)
    for p in range(world.dim_p):
        x = agent.state.p_pos[p]
        if x < -0.9 * wall_pos:
            raw_acc[p] = 1.0
            world.random_walk_buffer[2][p] = 1.0
            acc = acc_limit * raw_acc
            return acc
        elif x > 0.9 * wall_pos:
            raw_acc[p] = -1.0
            world.random_walk_buffer[2][p] = -1.0
            acc = acc_limit * raw_acc
            return acc
        else:
            pass

    # np.random.seed(world.script_agents_seed)
    
    if world.random_walk_buffer[0] == world.random_walk_buffer[1]:
        world.random_walk_buffer[2] = acc_limit * np.random.randn(world.dim_p)
        world.random_walk_buffer[0] = 0
        world.random_walk_buffer[1] = np.ceil(np.random.rand() * time_step_limit)

    if world.random_walk_buffer[0] <= world.random_walk_buffer[1]:
        acc = world.random_walk_buffer[2]
        world.random_walk_buffer[0] = world.random_walk_buffer[0] + 1



        # for p in range(world.dim_p):

        #     raw_acc = np.random.randn(world.dim_p)    
        #     acc = acc_limit * raw_acc
        #     world.curr_time_step = world.curr_time_step + 1
        # else:
        #     pass
        #     world.curr_time_step = 0
        #     world.total_time_step = np.random.rand() * time_step_limit
            

    # world.script_agents_seed = world.script_agents_seed + 1
    # agent_action.u = acc

    return acc    