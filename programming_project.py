#import mdp as tools
import numpy as np
import geometry as geom
from matplotlib import collections  as mc
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# INCORPORATE: DONT GET TOO CLOSE TO THE WALL!!
# INTERPOLATION!!


def is_outside_wall(x_width, y_width, curr_position):
    """
    determines whether a certain location is outside the room defined by x_width
    and y_width
    INPUT:
    x_width: Width in x direction of the room the bat is in
    y_width: Width in y direction of the room the bat is in
    curr_position: 2D array of the coordinates of the current position
    OUTPUT:
    boolean: False if position is inside the boundaries given by x_width and
    y_width, True if position is outside the boundaries
    """
    if (0.1 < curr_position[0] < x_width-0.1) and (0.1 < curr_position[1] < y_width-0.1): 
        return False
    else:
        return True

def random_walk(x_width, y_width, len_walk = 100):
    """
    INPUT:
    x_width: Width in x direction of the room the bat is in 
    y_width: Width in y direction of the room the bat is in
    resolution
    len_walk
    OUTPUT:
    positions:
    """
    #intialize positions vector, setting minimum speed and maximum speed
    positions      = np.zeros((len_walk+1,2))
    max_speed      = y_width/10.
    min_speed      = y_width/20.
    
    # intializing velocity vector
    speed_old    = np.random.uniform(min_speed,max_speed) 
    dir_old      = np.random.randint(0,2*np.pi)
    speed_new    = np.random.uniform(min_speed,max_speed) 
    dir_new      = np.random.randint(0,2*np.pi) 
    
    # computing the initial velocity vector
    velocity_old = speed_old*np.array([np.sin(dir_old),np.cos(dir_old)])
    
    # plaxing the bat in a random position in the room
    positions[0,0] = np.around(np.random.uniform(0.1,x_width-0.1))
    positions[0,1] = np.around(np.random.uniform(0.1,y_width-0.1))
 
    step = 1
    
    while step < len_walk:
        
        # compute (random) new velocity vector
        velocity_new = speed_new*np.array([np.sin(dir_new),np.cos(dir_new)])
        
        # interpolate between the old and the new velocity vector
        f = interp1d([velocity_old[0], velocity_new[0]], [velocity_old[1],velocity_new[1]])
        
        # take a value in the "interpolation range"
        velocity_x = np.mean([velocity_old[0],velocity_old[0],velocity_new[0]])
        velocity_new = np.array([velocity_x,f(velocity_x)])
        
        # take a step
        positions[step,:] = np.array([positions[step-1,0]+velocity_new[0],
                            positions[step-1,1]+velocity_new[1]])
                            
        if is_outside_wall(x_width, y_width, positions[step,:]):
            
            while is_outside_wall(x_width, y_width, positions[step,:]):
            
                #dist_to_wall = np.linalg.norm(positions[step-1,:]-positions[step,:])
                change_angle = dir_new+np.pi+np.random.uniform(-0.1*np.pi,0.1*np.pi)
                
                # bouncing back to at most 80% of the distance of the wall
                #slowing = np.random.uniform(0.1,0.8)
                #slowing = 2.
                #positions[step,:] = np.array([positions[step-1,0]+slowing*dist_to_wall*np.sin(change_angle),
                #                       positions[step-1,1]+slowing*dist_to_wall*np.cos(change_angle)])
                positions[step,:] = np.array([positions[step-1,0]+speed_new*np.sin(change_angle),
                                       positions[step-1,1]+speed_new*np.cos(change_angle)])
            step += 1
        else:
            step += 1
        
        velocity_old = velocity_new
        speed_new    = np.random.uniform(0.2,max_speed)
        dir_new      = np.random.randint(0,2*np.pi)
        
    positions = positions[0:-1,:]    
        
    return positions
    
    
def sense_the_walls_orthogonal(position, x_width = 5, y_width = 5, walls = 'ru'):
    """
    This function generates 2 sensors around the bad that are oriented orthogonal 
    to the walls and calculates the respecitve distances to the surrounding walls.

    INPUT:
    position: current position of the bat in a 2D array
    x_width: Width in x direction of the room the bat is in 
    y_width: Width in y direction of the room the bat is in
    
    OUTPUT:
    intersections: points on the walls where the vectors coming from the current
                   position in the ith sensors direction intersect the walls
    dists: distance to the walls (distance between the current location and 
            each respective intersection of the "sensor vectors" with the walls)
    """    
    N_sensors = 2
    x_pos = position[0]
    y_pos = position[1]
    
    intersections      = np.zeros([N_sensors,2])    
    intersections[:,0] = np.array([x_pos,y_width])
    intersections[:,1] = np.array([x_width,y_pos])
    
    dists    = np.zeros(2)
    dists[0] = y_width - y_pos
    dists[1] = x_width - x_pos
    
    return intersections, dists
    
def sense_the_walls(position, N_sensors = 4, x_width = 5, y_width = 5):
    """
    This function generates N sensors around the bad (in randomly distributed directions)
    and calculates the respecitve distances to the surrounding walls.

    INPUT:
    position: current position of the bat in a 2D array
    N_sensors: Number of sensors
    x_width: Width in x direction of the room the bat is in 
    y_width: Width in y direction of the room the bat is in
    
    OUTPUT:
    intersections: points on the walls where the vectors coming from the current
                   position in the ith sensors direction intersect the walls
    dists: distance to the walls (distance between the current location and 
            each respective intersection of the "sensor vectors" with the walls)
    """    
    
    # calculate the maximal distance to ensure that there will be an intersection
    # that can be detected for every snesor
    max_dist = np.linalg.norm(np.array([x_width, y_width]))
    
    # generate random directions for the sensors
    sensor_dirs = np.random.uniform(0,2*np.pi,N_sensors)
    
    # calculate the "endpoints" of the sensor vecotrs
    endpoints = np.dot(np.ones((N_sensors,1)), np.matrix(position))+max_dist*\
                np.matrix(np.array([np.sin(sensor_dirs),np.cos(sensor_dirs)])).T 
    
    # define the corners to ultimatively find the intersection points
    corners = np.array([[0.,0.],[0.,y_width],[x_width, y_width],[x_width, 0.]])
    
    # initialize matrix of intersection points
    intersections = np.zeros([N_sensors,2])
    
    # compute intersection points for all sensors
    for sensor in range(N_sensors):
        for edge in range(4):
            intersect = geom.getIntersectPoint(corners[edge-1,:], corners[edge,:],\
                        position, np.array(endpoints[sensor,:])[0])

            (start_x, end_x) = np.sort((endpoints[sensor,0],position[0]))           
            (start_y, end_y) = np.sort((endpoints[sensor,1],position[1]))
            
            if (start_x <= intersect[0][0] <= end_x) and (intersect[0][1]<= end_y):
                    if (0. <= intersect[0][0] <= x_width) and (0. <= intersect[0][1] <= y_width):
                        intersections[sensor,:] = intersect[0]
     

    # compute distances to the intersection points     
    dists = np.linalg.norm(np.dot(np.ones((N_sensors,1)), np.matrix(position))-intersections, axis = 1)  
    
    return intersections, dists
    


