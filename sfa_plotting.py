import sfa_bat as bat
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot') 

def plot_SF(sfa_flow, sensor_dirs, x_width, y_width, sensor_info = '', \
                     N_sensors = 4, orthogonal = False, grid_resolution = 10.):
    
    g_w, len_x, len_y = bat.grid_walk(x_width,y_width)
    inscts, grid_sense, sens_dirs = bat.sense_the_walls(g_w,x_width = x_width,\
                y_width = y_width, sensor_dirs = sensor_dirs, orthogonal = orthogonal)
    sf_grid = bat.back_to_grid(sfa_flow(grid_sense),len_x,len_y)    
    
    n_sf = min(np.shape(sf_grid)[2],8)    
        
#    if n_sf > 8:
#        features_vis = np.hstack((np.arange(0,3),np.arange(3,n_sf+1,int(round(n_sf-3)/4.))))[:8]

    plt.figure()
    plt.suptitle('Slow Features'+sensor_info)
    for sf in range(n_sf):
        
        
#        x = np.linspace(0,x_width,10*x_width)
#        y = np.linspace(0,y_width,10*y_width)
        
#        sfa_grid = np.zeros((len(x),len(y)))        
        
#        for i in range(len(x)):
#            for j in range(len(y)):
#                
#                gridpoint_ext = np.array([np.ones(len_rw)*x[i], np.ones(len_rw)*y[j]]).T               
#                dists = np.linalg.norm(gridpoint_ext-random_walk, axis = 1)
#                sfa_grid[i,j] = sfa_data[np.argmin(dists),sf]                
        
        
        x_ticks_loc = np.linspace(0,grid_resolution*x_width,x_width+1)
        y_ticks_loc = np.linspace(0,grid_resolution*y_width,y_width+1)
        
        plt.subplot(2,int(n_sf/2.),sf+1)
        plt.title('feature %i'%(sf+1))
        #plt.plot([0,x_width,x_width,0,0],[0,0,y_width,y_width,0],'k')
        plt.imshow(sf_grid[:,:,sf].T, origin = 'lower')
        #plt.scatter(random_walk[:,0],random_walk[:,1],c = sfa_data[:,sf])
        plt.xticks(x_ticks_loc,np.arange(0,x_width+1))
        plt.yticks(y_ticks_loc,np.arange(0,y_width+1))
        plt.xlabel('x position')
        plt.ylabel('y position')
    plt.show()
        
    
def plot_random_walk(random_walk, x_width, y_width ,title_info,\
                           color_code = True, trajectory = True):
    
    r_w_length = len(random_walk)
    
    plt.figure()
    plt.plot([0,x_width,x_width,0,0],[0,0,y_width,y_width,0],'k')
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