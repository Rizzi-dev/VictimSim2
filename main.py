import sys
import os
import time
import joblib  

from vs.environment import Env
from maze_explorer import Explorer
from maze_rescuer import Rescuer

def main(data_folder_name):
    current_folder = os.path.abspath(os.getcwd())
    data_folder = os.path.abspath(os.path.join(current_folder, data_folder_name))

    model_path = os.path.join(current_folder, "classificador_treinado.joblib")
    print(f"[MAIN] Loading model from: {model_path}")
    classifier = joblib.load(model_path)

    env = Env(data_folder)
    maze_width = Env(data_folder).dic['GRID_WIDTH']
    maze_height = Env(data_folder).dic['GRID_HEIGHT']

    rescuer_file = os.path.join(data_folder, "cfg_1", "rescuer_1_config.txt")
    master_rescuer = Rescuer(env, rescuer_file, 4)

    directions = [["down", "left"], ["left", "up"], ["right", "down"], ["up", "right"]]
    
    for i in range(1, 5):
        filename = f"explorer_{i:1d}_config.txt"
        explorer_file = os.path.join("datasets/data_430v_100x100/cfg_1", filename)
        Explorer(env, explorer_file, directions[i-1][0], directions[i-1][1], maze_width, maze_height, master_rescuer, classifier)

    env.run()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        data_folder_name = sys.argv[1]
    else:
        data_folder_name = os.path.join("datasets", "data_430v_100x100")
        
    main(data_folder_name)
