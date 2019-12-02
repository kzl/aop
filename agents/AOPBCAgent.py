from agents.BCAgent import BCAgent

class AOPBCAgent(BCAgent):
    """
    A BCAgent that uses the AOP planning mechanism.
    """

    def __init__(self, params):
        super(AOPBCAgent, self).__init__(params)
        self.has_aop = True
