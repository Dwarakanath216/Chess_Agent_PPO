from chess import Chess
from agents import SingleAgentChess, DoubleAgentsChess
from learnings.ppo import PPO

buffer_size = 32
if __name__ == "__main__":
    chess = Chess(window_size=512, max_steps=128, render_mode="rgb_array")
    chess.reset()

    ppo = PPO(
        chess,
        hidden_layers=(2048,) * 4,
        epochs=100,
        buffer_size=buffer_size * 2,
        batch_size=128,
    )

    print(ppo.device)
    print(ppo)
    print("-" * 64)

    # agent = DoubleAgentsChess(
    #     env=chess,
    #     learner=ppo,
    #     episodes=2000,
    #     train_on=buffer_size,
    #     result_folder="results/DoubleAgents",
    # )
    # agent.train(render_each=20, save_on_learn=True)
    # agent.save()
    # chess.close()

    agent = SingleAgentChess( 
    env=chess,
    learner=ppo,
    episodes=2000, # number of episodes to play/learn
    train_on=buffer_size, # current episode % train on == 0 then train
    result_folder="results/SingleAgent",
    )
    agent.train(
        render_each=20, # render and save the game into a episode_{n}.mp4 file
        save_on_learn=True # save the stats after each learning
    )
    agent.save()
    chess.close()
