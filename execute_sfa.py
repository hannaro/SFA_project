# file that tests functions in programming_project.py

import sfa_bat as bat
import sfa_analysis as analysis
import sfa_plotting as sfap
import matplotlib.pyplot as plt
plt.style.use('ggplot') 
import numpy as np

# After sufficiently many steps, did the bat (pretty much) visit every position?
x_width = 5
y_width = 5
r_w = bat.random_walk(x_width,y_width, 2000)
sfap.plot_random_walk(r_w,x_width,y_width,', Room Coverage',trajectory = False,\
                                                            color_code = True)

# plot trajectory

x_width = 3
y_width = 5
r_w = bat.random_walk(x_width,y_width, 500)
sfap.plot_random_walk(r_w,x_width,y_width,', Trajectory',trajectory = True,\
                                                            color_code = True)


### 3. orthogonal sensors

# a) rectangular room
x_width = 3
y_width = 5
r_w = bat.random_walk(x_width,y_width, len_walk = 10000)

intersects, sensory_data, sensor_dirs = bat.sense_the_walls(r_w, x_width = x_width,\
                                               y_width = y_width, orthogonal = True)                                                                                    
sfa_flow = analysis.sfa(sensory_data,x_width,y_width, N_sensors = 2,\
                                        orthogonal = True, poly_degree = 1)
sfap.plot_SF(sfa_flow, sensor_dirs, x_width, y_width, N_sensors = 2, orthogonal = True,\
                        sensor_info = ', orthogonal sensors, rectangular room')

# b) quadratic room

x_width = 5
y_width = 5
r_w = bat.random_walk(x_width,y_width, len_walk = 10000)

intersects, sensory_data, sensor_dirs = bat.sense_the_walls(r_w, x_width = x_width,\
                                               y_width = y_width, orthogonal = True)                                                                                    
sfa_flow = analysis.sfa(sensory_data,x_width,y_width, N_sensors = 2,\
                                        orthogonal = True, poly_degree = 1)
sfap.plot_SF(sfa_flow, sensor_dirs, x_width, y_width, N_sensors = 2, orthogonal = True,\
                        sensor_info = ', orthogonal sensors, rectangular room')



## 4. sensors in random directions


x_width = 5
y_width = 5
N_sensors = 4
r_w = bat.random_walk(x_width,y_width, len_walk = 10000)

intersects, sensory_data, sensor_dirs = bat.sense_the_walls(r_w, x_width = x_width, \
                                            y_width = y_width, N_sensors = N_sensors)
                                            
sfa_flow1 = analysis.sfa(sensory_data,x_width,y_width, N_sensors = N_sensors,\
                                                 poly_degree = 1, whitening = True)
sfap.plot_SF(sfa_flow1, sensor_dirs, x_width, y_width, N_sensors = N_sensors, \
                         sensor_info = ', %i random sensors, degree 1'%N_sensors)


## 5. increase polynomial degree to 3,5,7

sfa_flow3 = analysis.sfa(sensory_data,x_width,y_width, N_sensors = N_sensors,\
                                                 poly_degree = 3, whitening = True)
sfap.plot_SF(sfa_flow3, sensor_dirs, x_width, y_width, N_sensors = N_sensors, \
                         sensor_info = ', %i random sensors, degree 3'%N_sensors)
                         
sfa_flow5 = analysis.sfa(sensory_data,x_width,y_width, N_sensors = N_sensors,\
                                                 poly_degree = 5, whitening = True)
sfap.plot_SF(sfa_flow5, sensor_dirs, x_width, y_width, N_sensors = N_sensors, \
                         sensor_info = ', %i random sensors, degree 5'%N_sensors)
                         
sfa_flow7 = analysis.sfa(sensory_data,x_width,y_width, N_sensors = N_sensors,\
                                                 poly_degree = 7, whitening = True)
sfap.plot_SF(sfa_flow7, sensor_dirs, x_width, y_width, N_sensors = N_sensors, \
                         sensor_info = ', %i random sensors, degree 7'%N_sensors)



# strongly elongated room

x_width = 10
y_width = 1
N_sensors = 4
r_w = bat.random_walk(x_width,y_width, len_walk = 10000)

intersects, sensory_data, sensor_dirs = bat.sense_the_walls(r_w, x_width = x_width, \
                                            y_width = y_width, N_sensors = N_sensors)
                                            
flow_long = analysis.sfa(sensory_data,x_width,y_width, N_sensors = N_sensors,\
                                                 poly_degree = 3, whitening = True)
sfap.plot_SF(flow_long, sensor_dirs, x_width, y_width, N_sensors = N_sensors, \
                         sensor_info = ', elongated room')


## 7. ICA

# a) using all SFA outputs

x_width = 5
y_width = 5
N_sensors = 4
r_w = bat.random_walk(x_width,y_width, len_walk = 10000)

intersects, sensory_data, sensor_dirs = bat.sense_the_walls(r_w, x_width = x_width, \
                                            y_width = y_width, N_sensors = N_sensors)
                                            
flow_ica_all = analysis.sfa(sensory_data,x_width,y_width, N_sensors = N_sensors,\
                   poly_degree = 5, ica = True, out_dim = None, whitening = True)
sfap.plot_SF(flow_ica_all, sensor_dirs, x_width, y_width, N_sensors = N_sensors, \
                         sensor_info = ', ICA on all features')


# b) reduce number of output signals

J = 8
flow_ica_all = analysis.sfa(sensory_data,x_width,y_width, N_sensors = N_sensors,\
                   poly_degree = 5, ica = True, out_dim = J, whitening = True)
sfap.plot_SF(flow_ica_all, sensor_dirs, x_width, y_width, N_sensors = N_sensors, \
                         sensor_info = ', ICA on %i features'%J)



x_width = 5
y_width = 5
N_sensors = 4
r_w = bat.random_walk(x_width,y_width, len_walk = 10000)

intersects, sensory_data = bat.sense_the_walls(r_w, x_width = x_width, \
                                            y_width = y_width, N_sensors = N_sensors)
                                            
slow_ica_all = analysis.sfa(sensory_data, N_sensors = N_sensors, poly_degree = 5,\
                            whitening = True, ica = True, out_dim = None)
sfap.plot_SF(slow_ica_all, r_w, x_width,y_width,', ICA on all features')


###### plot the vectors from the current position to the walls with 4 random sensors   


#position = np.array([1,1])
#x_width = 2
#y_width = 5
#intersects, dists = bat.sense_the_walls_orthogonal(position,x_width,y_width) 

#(N,d) = np.shape(intersects)
#plt.figure(figsize = (5,5))
#plt.scatter(intersects[:,0],intersects[:,1])
## plot the current situation
#plt.plot(position[0],position[1],'or')
## plot one wall
#plt.plot(0.+np.linspace(0,1,100)*(x_width), np.zeros(100),'k')
#for i in range(N):
#    plt.plot(position[0]+np.linspace(0,1,100)*(intersects[i,0]-position[0]), \
#    position[1]+np.linspace(0,1,100)*(intersects[i,1]-position[1]),'b')
## plot other walls
#plt.plot(0.+np.linspace(0,1,100)*(x_width), np.ones(100)*y_width,'k')
#plt.plot(np.zeros(100), 0.+np.linspace(0,1,100)*(y_width),'k')
#plt.plot(np.ones(100)*y_width, 0.+np.linspace(0,1,100)*(y_width),'k')
#plt.xlim(-3.,x_width +3.)
#plt.ylim(-3.,y_width +3.)
#plt.xlabel('x')
#plt.ylabel('y')
#plt.legend(['current position of the bat','walls','sensor vectors'], loc = 'best')
#plt.show()
#print x_width

#plot distances over time
