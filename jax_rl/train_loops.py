import numpy as np

from buffers import ReplayBuffer
from dm_control_utils import eval_policy as eval_policy_dm_control
from dm_control_utils import flat_obs
from gym_utils import eval_policy as eval_policy_gym
from gym_utils import optional_squeeze


def dm_control_train_loop(args: dict, policy, replay_buffer: ReplayBuffer, env):
    evaluations = [
        eval_policy_dm_control(policy, args.domain_name, args.task_name, args.seed)
    ]

    timestep = env.reset()
    episode_reward = 0
    episode_timesteps = 0
    episode_num = args.load_step // 1000

    for t in range(args.load_step, int(args.max_timesteps)):

        episode_timesteps += 1

        state = flat_obs(timestep.observation)

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = np.random.uniform(
                env.action_spec().minimum,
                env.action_spec().maximum,
                size=env.action_spec().shape,
            )
        else:
            action = policy.select_action(state).clip(-args.max_action, args.max_action)

        # Perform action
        timestep = env.step(action)
        done_bool = float(timestep.last())

        # Store data in replay buffer
        replay_buffer.add(
            state, action, flat_obs(timestep.observation), timestep.reward, done_bool
        )

        episode_reward += timestep.reward

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            for _ in range(args.train_steps):
                if args.policy == "MPO":
                    policy = policy.update(
                        replay_buffer, args.batch_size, args.num_action_samples
                    )
                else:
                    policy = policy.train(replay_buffer, args.batch_size)

        if timestep.last():
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(
                f"Total T: {t+1} Episode Num: {episode_num+1} "
                f"Episode T: {episode_timesteps} Reward: {episode_reward:.3f}"
            )
            # Reset environment
            timestep = env.reset()
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            evaluations.append(
                eval_policy_dm_control(
                    policy, args.domain_name, args.task_name, args.seed
                )
            )
            np.save(f"./results/{args.file_name}_{t+1}", evaluations)
        # if (t + 1) % args.save_freq == 0:
        #     if args.save_model:
        #         policy.save(f"./models/{args.file_name}_{t+1}")


def gym_train_loop(args: dict, policy, replay_buffer: ReplayBuffer, env):
    evaluations = [eval_policy_gym(policy, args.env, args.seed)]

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = args.load_step // 1000

    for t in range(args.load_step, int(args.max_timesteps)):
        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = policy.select_action(state).clip(-args.max_action, args.max_action)
            action = optional_squeeze(action)

        # Perform action
        next_state, reward, done, _ = env.step(action)
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            for _ in range(args.train_steps):
                if args.policy == "MPO":
                    policy.train(
                        replay_buffer, args.batch_size, args.num_action_samples
                    )
                else:
                    policy.train(replay_buffer, args.batch_size)

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(
                f"Total T: {t+1} Episode Num: {episode_num+1} "
                f"Episode T: {episode_timesteps} Reward: {episode_reward:.3f}"
            )
            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            evaluations.append(eval_policy_gym(policy, args.env, args.seed))
            np.save(f"./results/{args.file_name}_{t+1}", evaluations)
        if (t + 1) % args.save_freq == 0:
            if args.save_model:
                policy.save(f"./models/{args.file_name}_{t+1}")
