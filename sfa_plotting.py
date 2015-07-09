import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot') 

def plot_SF(sfa_data,random_walk,sensor_info,n_sf = 2):
    
    for sf in range(n_sf):
        
        plt.figure()
        plt.title('Slow Feature %i'%(sf+1)+sensor_info)
        plt.scatter(random_walk[:,0],random_walk[:,1],c = sfa_data[:,sf])
        plt.xlabel('x position')
        plt.ylabel('y position')
        plt.show()
        
    
def plot_random_walk(random_walk, x_width, y_width ,title_info,\
                           color_code = True, trajectory = True):
    
    r_w_length = len(random_walk)
    
    plt.figure()
    if trajectory:
        plt.plot(random_walk[:,0],random_walk[:,1],'k',zorder = 1)
    if color_code:
        plt.scatter(random_walk[:,0],random_walk[:,1], c = np.arange(r_w_length),\
                                                                    zorder = 2)
    else:
        plt.scatter(random_walk[:,0],random_walk[:,1], zorder = 2)
    plt.xlabel('x position')
    plt.ylabel('y position')
    plt.title('Random Walk of length %i in %i x %i room'\
                                    %(r_w_length,x_width,y_width)+title_info)
    plt.show()