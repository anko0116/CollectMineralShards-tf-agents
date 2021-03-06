from tf_agents.trajectories import time_step

def compute_avg_return(environment, policy, num_episodes):
    total_return = 0.0
    for _ in range(num_episodes):
        print("INSIDE COMPUTE")    
        time_step = environment.reset()
        episode_return = 0.0

        while time_step["step_type"] != :
            print(time_step)
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
            total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]
