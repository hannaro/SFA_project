import numpy as np
import mdp


def sfa(sensory_input, N_sensors = 4, poly_degree = 3, whitening = False):

    if whitening:
        
        flow = (mdp.nodes.PolynomialExpansionNode(poly_degree) +
                mdp.nodes.WhiteningNode(svd=True,reduce = True) +
                mdp.nodes.SFANode(output_dim = N_sensors))
                
    else:

        flow = (mdp.nodes.PolynomialExpansionNode(poly_degree) +
                 mdp.nodes.SFANode(output_dim = N_sensors))
             
    flow.train(sensory_input)
    
    slow = flow(sensory_input)
    
    return slow