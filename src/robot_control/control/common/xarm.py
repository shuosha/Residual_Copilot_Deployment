XY_MIN, XY_MAX = 50, 800
X_MIN,X_MAX = 0, 800
Y_MIN,Y_MAX = -350, 350
Z_MIN, Z_MAX = 110, 650

MOVE_RATE = 1000  # hz; macro velocity = move_rate * xyz_velocity 
MOVE_SLEEP = 1 / MOVE_RATE
XYZ_VELOCITY = 1.0  # mm (in every MOVE_SLEEP interval)
ANGLE_VELOCITY_MAX = 0.05  # degree

GRIPPER_OPEN_MAX = 850
GRIPPER_OPEN_MIN = 0

POSITION_UPDATE_FREQ = 100.0  # command reader must read faster than this??
COMMAND_EXECUTION_FREQ = 100.0 # command sender must be faster than this

# velocity control
VELOCITY_DELTA_SCALE = 7.5

# position control
MAX_ACTIVATE_DELTA = 0.25
MAX_DELTA_NORM = 0.5