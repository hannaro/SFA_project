# file that tests functions in programming_project.py

from programming_project import *

# After sufficiently many steps, did the bat (pretty much) visit every position?
x_width = 5
y_width = 5
len_walk = 1000
r_w = random_walk(x_width,y_width, len_walk)
plt.figure()
plt.scatter(r_w[:,0],r_w[:,1])
plt.xlabel('x position')
plt.ylabel('y position')
plt.title('room coverage of random walk, room is %i x %i big and the bat took %i steps'%(x_width, y_width, len_walk))

# plot trajectory (with )
x_width = 5
y_width = 5
r_w = random_walk(x_width,y_width, len_walk = 20)
plt.figure()
plt.scatter(r_w[:,0],r_w[:,1],c=np.arange(len(r_w[:,0])))
plt.plot(r_w[:,0],r_w[:,1])
plt.xlim(0,x_width)
plt.ylim(0,y_width)
plt.xlabel('x position')
plt.ylabel('y position')
plt.title('trajectory')



# plot the vectors from the current position to the walls with 4 random sensors   
position = np.array([1,1])
intersects, dists = sense_the_walls(position)
(N,d) = np.shape(intersects)
plt.figure(figsize = (5,5))
plt.scatter(intersects[:,0],intersects[:,1])
# plot the current situation
plt.plot(position[0],position[1],'or')
# plot one wall
plt.plot(0.+np.linspace(0,1,100)*(x_width), np.zeros(100),'k')
for i in range(N):
    plt.plot(position[0]+np.linspace(0,1,100)*(intersects[i,0]-position[0]), \
    position[1]+np.linspace(0,1,100)*(intersects[i,1]-position[1]),'b')
# plot other walls
plt.plot(0.+np.linspace(0,1,100)*(x_width), np.ones(100)*y_width,'k')
plt.plot(np.zeros(100), 0.+np.linspace(0,1,100)*(y_width),'k')
plt.plot(np.ones(100)*y_width, 0.+np.linspace(0,1,100)*(y_width),'k')
plt.xlim(-3.,x_width +3.)
plt.ylim(-3.,y_width +3.)
plt.xlabel('x')
plt.ylabel('y')
plt.legend(['current position of the bat','walls','sensor vectors'], loc = 'best')
plt.show()


