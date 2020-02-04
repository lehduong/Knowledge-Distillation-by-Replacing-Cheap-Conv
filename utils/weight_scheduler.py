class WeightScheduler:
    def __init__(self, weight_groups):
        """
        :param weight_groups: a diction of dictionary. The passing parameters should have following format:
            {name: { value: float, anneal_rate: float}} e.g.
            { "alpha":{ "value": 0.1, "anneal_rate": 1.2, 'max': 1, 'min': 0},
              "beta": { "value": 0.2, "anneal_rate": 0.8, 'min': 0},
              "gamma": { "value": 1.5, "anneal_rate": 1}
            }
            name: str - indicate the name of that weight. The passed weight can be access as a attribute
                of WeightScheduler object
            value: float - the default value of that weight
            anneal_rate: float - ratio which the weight will be multiplied at each step
        """
        self.weights = weight_groups

    def step(self):
        for k in self.weights.keys():
            weight_info = self.weights[k]
            updated_value = weight_info['value']*weight_info['anneal_rate']
            if 'min' in weight_info:
                updated_value = updated_value if updated_value > weight_info['min'] else weight_info['min']
            if 'max' in weight_info:
                updated_value = updated_value if updated_value < weight_info['max'] else weight_info['max']
            self.weights[k]['value'] = updated_value

    def __getattr__(self, name):
        return self.weights[name]['value']
