import sys
import os
import time

from vs.environment import Env
from maze_explorer import Explorer
from maze_rescuer import Rescuer

def main(data_folder_name):
    current_folder = os.path.abspath(os.getcwd())
    data_folder = os.path.abspath(os.path.join(current_folder, data_folder_name))

    env = Env(data_folder)
    maze_width = Env(data_folder).dic['GRID_WIDTH']
    maze_height = Env(data_folder).dic['GRID_HEIGHT']

    rescuer_file = os.path.join(data_folder, "rescuer_1_config.txt")
    master_rescuer = Rescuer(env, rescuer_file, 4)

    filename = f"explorer_config.txt"
    explorer_file = os.path.join(data_folder, filename)
    
    directions = [["down", "left"], ["left", "up"], ["right", "down"], ["up", "right"]]
    
    for i in range(1, 5):
        filename = f"explorer_{i:1d}_config.txt"
        explorer_file = os.path.join(data_folder, filename)
        Explorer(env, explorer_file, directions[i-1][0], directions[i-1][1], maze_width, maze_height, master_rescuer)

    env.run()
    
        
if __name__ == '__main__':
    """ To get data from a different folder than the default called data
    pass it by the argument line"""
    
    if len(sys.argv) > 1:
        data_folder_name = sys.argv[1]
    else:
        data_folder_name = os.path.join("datasets", "data_430v_94x94")
        
    main(data_folder_name)
