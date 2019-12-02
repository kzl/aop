import numpy as np

class ReplayBuffer():
	"""
	Simple replay buffer, keeps trajectories together when accessed with
	modular arithmetic.
	"""

	def __init__(self, state_dim, act_dim, size, z_dim=0):
		self.buffer = dict()

		self.buffer['s'] = np.zeros((size, state_dim))
		self.buffer['a'] = np.zeros((size, act_dim))
		self.buffer['r'] = np.zeros(size)
		self.buffer['ns'] = np.zeros((size, state_dim))
		self.buffer['d'] = np.zeros(size)
		self.buffer['z'] = np.zeros((size, z_dim))

		self.ind = 0
		self.total_in = 0
		self.size = size
		self.N = state_dim
		self.M = act_dim
		self.z_dim = z_dim

	def update(self, s, ns, r, a, done=0, z=0):
		self.buffer['s'][self.ind] = s
		self.buffer['ns'][self.ind] = ns
		self.buffer['r'][self.ind] = r
		self.buffer['a'][self.ind] = a
		self.buffer['d'][self.ind] = done # 1 if done else 0
		if self.z_dim > 0:
			self.buffer['z'][self.ind] = z
		
		self.advance()

	def get(self, query, ind):
		return self.buffer[query][ind % self.size]

	def advance(self):
		self.ind = (self.ind+1) % self.size
		self.total_in += 1

	def get_size(self):
		return min(self.size, self.total_in)

	def sample(self):
		ind = random.randint(0, self.get_size()-1)
		s = self.get('s', ind)
		ns = self.get('ns', ind)
		a = self.get('a', ind)
		r = self.get('r', ind)
		z = self.get('z', ind)
		return s, ns, a, r, z

	def sample_seq(self, seq_len=1):
		ind = random.randint(0, self.get_size()-seq_len)

		s = np.zeros((seq_len, self.N))
		ns = np.zeros((seq_len, self.N))
		r = np.zeros(seq_len)
		a = np.zeros((seq_len, self.M))
		z = np.zeros((seq_len, self.z_dim))

		for i in range(seq_len):
			s[i] = self.get('s', ind+i)
			ns[i] = self.get('ns', ind+i)
			r[i] = self.get('r', ind+i)
			a[i] = self.get('a', ind+i)
			z[i] = self.get('z', ind+i)

		return s, ns, a, r, z
