# meu_teste/main.py (Versão Final com Nomes Corretos)

import sys
import os
from pathlib import Path

sys.path.append(str(Path.cwd()))

from vs.environment import Env
# Importa as classes com os nomes corretos dos arquivos renomeados
from agents.explorer import Explorer
from agents.rescuer import Rescuer

def main(scenario_folder_path, explorer_config, rescuer_config):
    """
    Função principal que agora usa os nomes de classe exatos 'Explorer' e 'Rescuer'.
    """
    print(f"Carregando ambiente da pasta: {scenario_folder_path}")
    env = Env(scenario_folder_path)
    
    print(f"Carregando agente explorador com a configuração: {explorer_config}")
    # Instancia a classe Explorer
    explorer_agent = Explorer(env, explorer_config)
    
    print(f"Carregando agente socorrista com a configuração: {rescuer_config}")
    # Instancia a classe Rescuer
    rescuer_agent = Rescuer(env, rescuer_config)
    
    print("\nIniciando a simulação...")
    env.run()
    print("Simulação finalizada.")
    
if __name__ == '__main__':
    scenario_folder = "datasets/data_10v_12x12"
    
    explorer_config_file = "agents/explorer_config.txt"
    rescuer_config_file = "agents/explorer_config.txt" 

    if not os.path.isdir(scenario_folder):
        print(f"ERRO: Pasta de cenário não encontrada em '{scenario_folder}'")
    elif not os.path.exists(explorer_config_file):
        print(f"ERRO: Arquivo de configuração do explorador não encontrado.")
    else:
        main(scenario_folder, explorer_config_file, rescuer_config_file)