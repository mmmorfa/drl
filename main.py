from custom_env import CustomEnv
from DQN_agent import DQNAgent
import numpy as np
from custom_env import NUM_ACTIONS, EPISODES, OBSERVATION_SPACE_SIZE, BATCH_SIZE


# Crear una instancia del entorno personalizado
env = CustomEnv()

# Crear una instancia del agente DQN
agent = DQNAgent(NUM_ACTIONS)


# Entrenamiento del agente
for episode in range(1, EPISODES + 1):
    state = env.reset()
    state = np.reshape(state, [1, OBSERVATION_SPACE_SIZE])
    done = False
    total_reward = 0

    while not done:
        # Elegir una acción usando el agente DQN
        action = agent.act(state)

        # Ejecutar la acción en el entorno
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, OBSERVATION_SPACE_SIZE])

        # Almacenar la experiencia en la memoria del agente
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        # Actualizar el modelo del agente
        agent.replay(BATCH_SIZE)

    # Imprimir información sobre el episodio
    print(f"Episodio: {episode}, Recompensa Total: {total_reward}, Exploración: {agent.epsilon}")

# Probar el agente entrenado
test_episodes = 10
total_rewards = []

for _ in range(test_episodes):
    state = env.reset()
    state = np.reshape(state, [1, OBSERVATION_SPACE_SIZE])
    done = False
    episode_reward = 0

    while not done:
        # Elegir la mejor acción según el modelo entrenado
        action = np.argmax(agent.model.predict(state)[0])

        # Ejecutar la acción en el entorno
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, OBSERVATION_SPACE_SIZE])

        state = next_state
        episode_reward += reward

    total_rewards.append(episode_reward)

# Calcular y mostrar la recompensa promedio en los episodios de prueba
average_reward = sum(total_rewards) / test_episodes
print(f"Recompensa Promedio en {test_episodes} Episodios de Prueba: {average_reward}")
