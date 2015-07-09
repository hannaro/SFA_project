import numpy as np
import mdp


def sfa(sensory_input, N_sensors = 4, poly_degree = 3, whitening = False, ica = False,
                                                        out_dim = None):

    if whitening:
        
        flow = (mdp.nodes.PolynomialExpansionNode(poly_degree) +
                mdp.nodes.WhiteningNode(svd=True,reduce = True) +
                mdp.nodes.SFANode())
                
    else:

        flow = (mdp.nodes.PolynomialExpansionNode(poly_degree) +
                 mdp.nodes.SFANode())
                 
    if ica:
        
        flow = (mdp.nodes.PolynomialExpansionNode(poly_degree) +
                mdp.nodes.WhiteningNode(svd=True,reduce = True) +
                mdp.nodes.SFANode(output_dim = out_dim) +
                mdp.nodes.CuBICANode())
        
             
    flow.train(sensory_input)
    
    slow = flow(sensory_input)
    
    return slow