import numpy as np

class AliasMethod:
    """
    Implementation of the Vose Alias Method which should have a O(n) init
    instead of O(nlogn)

    Following https://www.keithschwarz.com/darts-dice-coins/
    """
    
    def __init__(self, population, p):
        self.population = population
        self.p = p

        assert population.shape[0] == p.shape[0], print("Population and probabilities should be of the same size.")
        # maybe a check for probability summing to 1 would be nice

        # initialize the algorithm

        self.n = population.shape[0]
        self.alias = np.zeros(self.n, dtype=np.int)
        self.prob = np.zeros(self.n)

        small = []
        large = []

        self.scaled_p = np.zeros(self.n)
        for i in range(self.n):
            self.scaled_p[i] = self.p[i] * self.n

            if self.scaled_p[i] < 1:
                small.append(i)
            else:
                large.append(i)
        

        while len(large) > 0 and len(small) > 0:
            l = small.pop(0)
            g = large.pop(0)
            self.prob[l] = self.scaled_p[l]
            self.alias[l] = g

            self.scaled_p[g] = (self.scaled_p[g] + self.scaled_p[l]) - 1

            if self.scaled_p[g] < 1:
                small.append(g)
            else:
                large.append(g)
        
        while len(large) > 0:
            self.prob[large.pop(0)] = 1

        while len(small) > 0:
            self.prob[small.pop(0)] = 1

    def sample(self, n=1):
        i = np.random.randint(0, self.n, size=(n))
        dice = np.random.uniform(0, 1, size=(n))
        filtered = np.where(self.prob[i] > dice, self.population[i], self.population[self.alias[i]])
        if filtered.shape[0] == 1:
            return filtered[0]
        
        return filtered
                

def node_sample(v_i, v_j, alias, size=5):
    sample_list = np.zeros((v_i.shape[0], size))
    for i in range(v_i.shape[0]):
        source = v_i[i]
        dest = v_j[i]
        samples = []
        while len(samples) < size:
            node = alias.sample()
            if node == source or node == dest:
                continue
            samples.append(node)
        sample_list[i] = samples
    
    return sample_list
        


if __name__ == "__main__":
    data = np.array([1, 2, 3])
    probs = np.array([0.5, 0.4, 0.1])
    sampler = AliasMethod(data, probs)
    print("===Testing single sample===")
    for i in range(10):
        num = sampler.sample()
        print(num)

    print("===Testing sample n = 100 ===")
    sample = sampler.sample(100)
    num, counts = np.unique(sample, return_counts=True)
    print(f"Numbers {num} with counts {counts}")