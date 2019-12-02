from agents.TD3pAgent import TD3pAgent

class AOPTD3Agent(TD3pAgent):
    """
    A TD3pAgent that uses the AOP planning mechanism.
    """

    def __init__(self, params):
        super(AOPTD3Agent, self).__init__(params)
        self.has_aop = True
