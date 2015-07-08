#import mdp as tools
import numpy as np
import geometry as geom
from matplotlib import collections  as mc
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# INCORPORATE: DONT GET TOO CLOSE TO THE WALL!!
# INTERPOLATION!!


def is_outside_wall(x_width, y_width, curr_position, wall_tol = 0.1):
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
    if ((wall_tol < curr_position[0] < x_width-wall_tol) \
                        and (wall_tol < curr_position[1] < y_width-wall_tol)): 
        return False
    else:
        return True

def random_walk(x_width, y_width, len_walk = 100, weight_smooth = 0.7):
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
    max_speed      = (y_width*x_width)/5.
    min_speed      = (y_width*x_width)/10.
    
    # intializing velocity vector
    speed_old    = np.random.uniform(min_speed,max_speed) 
    dir_old      = np.random.uniform(0,2*np.pi)
    speed_new    = np.random.uniform(min_speed,max_speed) 
    dir_new      = np.random.uniform(0,2*np.pi) 
    
    # computing the initial velocity vector
    velocity_old = speed_old*np.array([np.sin(dir_old),np.cos(dir_old)])
    
    # plaxing the bat in a random position in the room
    positions[0,0] = np.random.uniform(0.1,x_width-0.1)
    positions[0,1] = np.random.uniform(0.1,y_width-0.1)
 
    step = 1
    
    while step < len_walk:
        
        # compute (random) new velocity vector
        velocity_new = speed_new*np.array([np.sin(dir_new),np.cos(dir_new)])
        
        # interpolate between the old and the new velocity vector
        f = interp1d([velocity_old[0], velocity_new[0]], [velocity_old[1],velocity_new[1]])
        
        # take a value in the "interpolation range"
        velocity_x = weight_smooth*velocity_old[0]+(1-weight_smooth)*velocity_new[0]
        velocity_new = np.array([velocity_x,f(velocity_x)])
        
        # take a step
        positions[step,:] = np.array([positions[step-1,0]+velocity_new[0],
                            positions[step-1,1]+velocity_new[1]])


        if is_outside_wall(x_width, y_width, positions[step,:]):  ## while?!

            pos_inwall = positions[step-1,:]
            pos_outwall = positions[step,:]

            intersection, wall_type = get_wall_intersect(pos_inwall, pos_outwall,\
                            x_width,y_width)  
            valid_position = bounce_off_wall(intersection,pos_inwall,wall_type,\
                                                               x_width, y_width)
                        
            positions[step,:] = valid_position
            
        step += 1
        
        velocity_old = velocity_new
        speed_new    = np.random.uniform(min_speed,max_speed)
        dir_new      = np.random.uniform(0,2*np.pi)
        
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
    endpoints = np.array(np.dot(np.ones((N_sensors,1)), np.matrix(position))+max_dist*\
                np.matrix([np.sin(sensor_dirs),np.cos(sensor_dirs)]).T) 
    
    # initialize matrix of intersection points
    intersections = np.zeros([N_sensors,2])
    
    # compute intersection points for all sensors
    for sensor in range(N_sensors):    
        intersections[sensor,:],w_type = get_wall_intersect(position,endpoints[sensor,:],\
                                                                        x_width,y_width)

    # compute distances to the intersection points     
    dists = np.linalg.norm(np.dot(np.ones((N_sensors,1)), np.matrix(position))-intersections, axis = 1)  
    
    return intersections, dists
    
    
def get_wall_intersect(position, endpoint, x_width, y_width):
    
    # define the corners to ultimatively find the intersection points
    corners = np.array([[0.,0.],[0.,y_width],[x_width, y_width],[x_width, 0.]])    
    
    for edge in range(4):
        intersect = geom.getIntersectPoint(corners[edge-1,:], corners[edge,:],\
                    position, endpoint)[0]

        (start_x, end_x) = np.sort((endpoint[0],position[0]))           
        (start_y, end_y) = np.sort((endpoint[1],position[1]))
        
        if (start_x <= intersect[0] <= end_x) and (start_y <= intersect[1]<= end_y):
                if (0. <= intersect[0] <= x_width) and (0. <= intersect[1] <= y_width):
                    intersection = intersect
                    
    if intersection[0] in [0,x_width]:
        wall_type = 'vertical'
    else:
        wall_type = 'horizontal'                 
                    
    return intersection, wall_type
    

def bounce_off_wall(intersect,position,wall_type, x_width, y_width,wall_tol = 0.1 ):
    
    if wall_type == 'vertical':
        
        # get the vertical difference and the sign
        v_diff = intersect[1]-position[1]
        sign_v_diff = np.sign(v_diff)
        
        # determine distance to the wall we hit
        if sign_v_diff < 0:
            dist_to_wall = position[1]
        else:
            dist_to_wall = y_width - position[1]
        
        # walk parallel to wall (einfallswinkel = ausfallswinkel) until you hit next wall
        walk_y = min([dist_to_wall-wall_tol, 2*abs(v_diff)])
        pos_valid = np.array([position[0], position[1] + sign_v_diff*walk_y ])
        
    if wall_type == 'horizontal':
        
        # get horizontal difference
        v_diff = intersect[0]-position[0]
        sign_v_diff = np.sign(v_diff)
        
        # determine distance to the wall we hit
        if sign_v_diff < 0:
            dist_to_wall = position[0]
        else:
            dist_to_wall = x_width - position[0]
            
        # walk parallel to wall unless we hit the next
        walk_x = min([dist_to_wall-wall_tol, 2*abs(v_diff)])
        pos_valid = np.array([position[0] + sign_v_diff*walk_x, position[1]])
        
    return pos_valid
            