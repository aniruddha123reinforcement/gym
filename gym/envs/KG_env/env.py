
from math import inf
from typing import Optional
import numpy as np
import gym 
from gym import spaces
from gym import utils 
import random
from utils import *


class Knowledgegraph_gym(gym.Env):
	"""knowledge graph environment definition"""
	def __init__(self, dataPath, task=None):
	
		f1 = open(dataPath + 'entity2id.txt')
		f2 = open(dataPath + 'relation2id.txt')
		self.entity2id = f1.readlines()
		self.relation2id = f2.readlines()
		f1.close()
		f2.close()
		self.entity2id_ = {}
		self.relation2id_ = {}
		self.relations = []
		for line in self.entity2id:
			self.entity2id_[line.split()[0]] =int(line.split()[1])
		for line in self.relation2id:
			self.relation2id_[line.split()[0]] = int(line.split()[1])
			self.relations.append(line.split()[0])
		self.entity2vec = np.loadtxt(dataPath + 'entity2vec.bern')
		self.relation2vec = np.loadtxt(dataPath + 'relation2vec.bern')


		self.path = []
		self.path_relations = []


		# Knowledge Graph for path finding
		f = open(dataPath + 'kb_env_rl.txt')
		kb_all = f.readlines()
		f.close()
                
		self.kb = []
		if task != None:
			relation = task.split()[2]
			for line in kb_all:
				rel = line.split()[2]
				if rel != relation and rel != relation + '_inv':
					self.kb.append(line)

		self.die = 0 # record how many times does the agent choose an invalid path
		self.observation_space = spaces.Box(low=-inf, high=inf, shape=(200,), dtype=np.float32)
		self.action_space = spaces.Discrete(400)
		self.state = None
		self.steps_beyond_terminated = None
		
	def idx_state(self, idx_list):
		if idx_list != None:
			curr = self.entity2vec[idx_list[0],:]
			targ = self.entity2vec[idx_list[1],:]
			return np.expand_dims(np.concatenate((curr, targ - curr)),axis=0)
		else:
			return None	
		
	def reset(self, *,dataPath,kb,kb_inv, seed: Optional[int] = None,options: Optional[dict] = None):
		super.reset(seed=seed)
		#Call env.make before this 
		#Call gym.make(datapath,train[i_episode])
		f = open(dataPath + '/train.pairs')
		train_data = f.readlines()
		f.close()
		train_pairs = []
		for line in train_data:
			e1 = line.split(',')[0].replace('thing$','')
			e2 = line.split(',')[1].split(':')[0].replace('thing$','')
			if (e1 not in kb.entities) or (e2 not in kb.entities):
				continue
			train_pairs.append((e1,e2))
			# label = 1 if line[-2] == '+' else 0
			# sample = train_pairs.split()
			# state_idx = [env.entity2id_[sample[0]], env.entity2id_[sample[1]], 0]
			state_idx = [self.entity2id_[e1], self.entity2id_[e2], 0]
			self.state = self.idx_state(state_idx)
			self.steps_beyond_terminated = None
		return np.array(self.state,dtype = np.float32),{}
	

	def step(self, action):
		assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid"
		assert self.state is not None, "Call reset before using step method."
		terminated = 0 # Whether the episode has finished
		curr_pos, target_pos = 0,self.state
		chosed_relation = self.relations[action]
		choices = []
		for line in self.kb:
			triple = line.rsplit()
			e1_idx = self.entity2id_[triple[0]]
			
			if curr_pos == e1_idx and triple[2] == chosed_relation and triple[1] in self.entity2id_:
				choices.append(triple)
		if len(choices) == 0:
			reward = -1
			self.die += 1
			self.state = (curr_pos, target_pos, self.die) # stay in the initial state
			return np.array(self.state,dtype=np.float32),reward, terminated, False, {}
		else: # find a valid step
			path = random.choice(choices)
			self.path.append(path[2] + ' -> ' + path[1])
			self.path_relations.append(path[2])
			# print('Find a valid step', path)
			# print('Action index', action)
			self.die = 0
			new_pos = self.entity2id_[path[1]]
			reward = 0
			self.state = (new_pos, target_pos, self.die)

			if new_pos == target_pos:
				print('Find a path:', self.path)
				terminated = 1
				reward = 1
				self.state = None
			return np.array(self.state,dtype=np.float32),reward, terminated, False, {}


	def get_valid_actions(self, entityID):
		actions = set()
		for line in self.kb:
			triple = line.split()
			e1_idx = self.entity2id_[triple[0]]
			if e1_idx == entityID:
				actions.add(self.relation2id_[triple[2]])
		return np.array(list(actions))

	def path_embedding(self, path):
		embeddings = [self.relation2vec[self.relation2id_[relation],:] for relation in path]
		embeddings = np.reshape(embeddings, (-1,embedding_dim))
		path_encoding = np.sum(embeddings, axis=0)
		return np.reshape(path_encoding,(-1, embedding_dim))


