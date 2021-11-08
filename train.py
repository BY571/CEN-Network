
import gym
import pybullet_envs
import numpy as np
from collections import deque
import torch
import wandb
import argparse
import glob
import random
from agents import pg_agent
from networks import cen_networks
from buffer import onpolicy_buffer
from utils_ import utils


def get_config():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument("--run_name", type=str, default="CEN-discrete", help="Run name, default: CEN")
    parser.add_argument("--env", type=str, default="CartPole-v0", help="Gym environment name, default: Pendulum-v0")
    parser.add_argument("--episodes", type=int, default=300, help="Number of episodes, default: 100")
    parser.add_argument("--seed", type=int, default=1, help="Seed, default: 1")
    parser.add_argument("--log_video", type=int, default=0, help="Log agent behaviour to wanbd when set to 1, default: 0")
    parser.add_argument("--save_every", type=int, default=100, help="Saves the network every x epochs, default: 25")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size, default: 256")
    parser.add_argument("--eval_every", type=int, default= 5, help="s")
    # CEN Parameter
    parser.add_argument("--cen_hidden", type=int, default=32, help="")
    parser.add_argument("--lr", type=float, default=1e-4, help="")
    parser.add_argument("--ensemble_size", type=int, default=1, help="")
    parser.add_argument("--prediction_type", type=str, default="deterministic", choices=["deterministic", "probabilistic"], help="")
    parser.add_argument("--alpha", type=float, default=0.01, help="")
    parser.add_argument("--train_cen", type=int, default=1, help="")
    parser.add_argument("--reward_type", type=str, default="intrinsic", choices=["real", "intrinsic", "sparse"], help="")
    
    args = parser.parse_args()
    return args

def create_rewards(config, rewards: torch.Tensor, intrinsic_rewards:torch.Tensor)-> torch.Tensor:
    if config.reward_type == "real" and not config.train_cen:
        rewards = rewards
    elif config.reward_type == "real" and config.train_cen:
        rewards = rewards + intrinsic_rewards
    elif config.reward_type == "intrinsic" and config.train_cen:
        rewards = intrinsic_rewards
    elif config.reward_type == "sparse":
        rewards = torch.zeros(len(rewards))
    else:
        raise ValueError
    #print("rewards-shape", rewards.shape)
    return rewards


def train(config):
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    env = gym.make(config.env)
    eval_env = gym.make(config.env)
    
    env.seed(config.seed)
    env.action_space.seed(config.seed)

    eval_env.seed(config.seed)
    eval_env.action_space.seed(config.seed)

    observation_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    steps = 0
    average10 = deque(maxlen=10)
    
    with wandb.init(project="CEN", name=config.run_name, config=config):
        
        agent = pg_agent.Policy(observation_space=observation_size,
                                action_space=action_size,
                                ).to(device)

        cen_network = cen_networks.CENNetwork(observation_space=observation_size,
                                              action_space=action_size,
                                              lr=config.lr,
                                              hidden_size=config.cen_hidden,
                                              prediction_type=config.prediction_type,
                                              ensemble_size=config.ensemble_size,
                                              output_size=observation_size,
                                              alpha=config.alpha)
        
        buffer = onpolicy_buffer.OnpolicyBuffer(device=device)
        wandb.watch(agent, log="gradients", log_freq=10)
        
        #collect_random(env=env, dataset=buffer, num_samples=500)
        
        if config.log_video:
            env = gym.wrappers.Monitor(env, './video', video_callable=lambda x: x%10==0, force=True)

        for i in range(1, config.episodes+1):
            state = env.reset()
            episode_steps = 0
            while True:
                action, logprob = agent.get_action(torch.from_numpy(state).float().to(device))
                steps += 1
                next_state, reward, done, _ = env.step(action)
                buffer.add(state, action, reward, next_state, done, logprob)
                state = next_state
                episode_steps += 1
                if done:
                    break
            
            # Training
            # (states, actions, rewards, next_states, dones, logprobs)
            samples = buffer.sample()
            # Train CEN
            if config.train_cen:
                intrinsic_rewards, cen_loss = cen_network.train(observation=samples[0],
                                                                action=samples[1],
                                                                next_observation=samples[3])
                
                intrinsic_rewards = intrinsic_rewards.sum(1, keepdim=True).detach().squeeze()
            else:
                intrinsic_rewards = torch.zeros(samples[2].shape)
                cen_loss = {"CEN loss": 0}
            rewards = create_rewards(config, samples[2], intrinsic_rewards)
            
            # Train agent
            log_dict = agent.train(dones=samples[-2], rewards=rewards, log_probs=samples[-1])
            episode_logs = {"Epoch": i, "Episode length": episode_steps}
            log_dict.update(episode_logs)
            log_dict.update(cen_loss)
            wandb.log(log_dict, step=steps)

            # Evaluation
            if i % config.eval_every == 0:
                test_rewards = utils.evaluate(eval_env, agent, device, eval_runs=5)
                average10.append(test_rewards)
                wandb.log({"Epoch": i,
                           "Rewards": test_rewards,
                           "Average10": np.mean(average10)}, step=steps)
                print("Episode: {} | Reward: {}".format(i, test_rewards))

            if (i %10 == 0) and config.log_video:
                mp4list = glob.glob('video/*.mp4')
                if len(mp4list) > 1:
                    mp4 = mp4list[-2]
                    wandb.log({"gameplays": wandb.Video(mp4, caption='episode: '+str(i-10), fps=4, format="gif"), "Episode": i})

            if i % config.save_every == 0:
                utils.save(config, save_name="CAN", model=agent.policy, wandb=wandb, ep=0)

if __name__ == "__main__":
    config = get_config()
    train(config)
