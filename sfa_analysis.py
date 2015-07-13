import numpy as np
import mdp


def sfa(sensory_input, x_width, y_width, N_sensors = 4, poly_degree = 1, \
         whitening = False, grid_resolution = 10.,ica = False, out_dim = None,
         orthogonal = False):

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
    
    return flow