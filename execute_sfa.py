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



######### task 3 ##########

# do long random walk and plot

x_width = 5
y_width = 5
r_w = bat.random_walk(x_width,y_width, len_walk = 10000)

intersects, sensory_data = bat.sense_the_walls(r_w, x_width = x_width, \
                                            y_width = y_width)
slow = analysis.sfa(sensory_data, N_sensors = 4, poly_degree = 3, whitening = True)
sfap.plot_SF(slow, r_w, ', orthogonal sensors',n_sf = 4)






###### plot the vectors from the current position to the walls with 4 random sensors   


#position = np.array([1,1])
#x_width = 2
#y_width = 5
#intersects, dists = bat.sense_the_walls_orthogonal(position,x_width,y_width) #sense_the_walls(position)

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
