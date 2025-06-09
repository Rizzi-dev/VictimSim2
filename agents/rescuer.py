# agents/rescuer.py

from vs.abstract_agent import AbstAgent

class Rescuer(AbstAgent):
    """
    Um agente socorrista 'dummy' (fantoche) que não faz nada.
    Sua única finalidade é ser instanciado para que o simulador
    possa iniciar o loop principal.
    """
    def __init__(self, env, config_file):
        super().__init__(env, config_file)

    def deliberate(self) -> str:
        # Este agente fica parado e não faz nenhuma ação.
        return "no-op"