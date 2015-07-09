import numpy as np
import mdp


def sfa(sensory_input,sensors,poly_degree):

    flow = (mdp.nodes.PolynomialExpansionNode(poly_degree) +
             mdp.nodes.SFANode())
             
    flow.train(sensory_input)
    
    slow = flow(sensory_input)
    
    return slow