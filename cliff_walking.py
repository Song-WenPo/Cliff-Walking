import matplotlib.pyplot as plt
import numpy as np
from numpy import random

ALPHA = 0.1
EPSILON = 0.1
GAMMA = 1
COL = 12
ROW = 4
action_list = ['up', 'down', 'left', 'right']

def move_reward(x, y, action):
    (X, Y) = (x, y)
    if 0 < X < 11 and Y == 1 and action == 'down':
        X = 0
        Y = 0
        return X, Y, -100
    if X == 0 and Y == 0 and action == 'right':
        X = 0
        Y = 0
        return X, Y, -100
    if (action == 'up') and (Y < ROW - 1):
        Y += 1
    elif(action == 'left') and (X > 0):
        X -= 1
    elif(action == 'right') and (X < COL - 1):
        X += 1
    elif(action == 'down') and (Y > 0):
        Y -= 1
    return X, Y, -1

def get_state(x, y):
    return y * COL + x

def epsilon_greedy(x, y, q, epsilon):
    rannum = random.random()
    state = get_state(x, y)
    if rannum < epsilon:
        action = action_list[random.choice(4)]
    else:
        action = action_list[np.argmax(q[:, state])]
    return action

def MaxQ(state, q):
    max_action = action_list[np.argmax(q[:, state])]
    return max_action

#===============================#
# Q_learning and Sarsa function #
#===============================#

def Q_learning(q):
    runs = 30
    rewards = np.zeros([500])
    for j in range(runs):
        for i in range(500):
            rewards_sum = 0
            x = 0
            y = 0
            while True:
                action = epsilon_greedy(x, y, q, EPSILON)
                action_index = action_list.index(action)
                x_next, y_next, reward = move_reward(x, y, action)
                rewards_sum += reward
                state = get_state(x, y)
                next_state = get_state(x_next, y_next)
                max_action = MaxQ(next_state, q)
                max_action_index = action_list.index(max_action)
                q[action_index, state] +=  ALPHA * (reward + GAMMA * q[max_action_index, next_state] - q[action_index, state])
                if x == COL - 1 and y == 0:
                    break
                x = x_next
                y = y_next
            rewards[i] += rewards_sum
    rewards /= runs
    avg_rewards = []
    for i in range(9):
        avg_rewards.append(np.mean(rewards[:i+1]))
    for i in range(10,len(rewards)+1):
        avg_rewards.append(np.mean(rewards[i-10:i]))
    return avg_rewards
    

def Sarsa(q):
    runs = 30
    rewards = np.zeros([500])
    for j in range(runs):
        for i in range(500):
            rewards_sum = 0
            x = 0
            y = 0
            action = epsilon_greedy(x, y, q, EPSILON)
            while True:
                action_index = action_list.index(action)
                x_next, y_next, reward = move_reward(x, y, action)
                rewards_sum += reward
                state = get_state(x, y)
                next_state = get_state(x_next, y_next)
                max_action = epsilon_greedy(x_next, y_next, q, EPSILON)
                max_action_index = action_list.index(max_action)
                q[action_index, state] +=  ALPHA * (reward + GAMMA * q[max_action_index, next_state] - q[action_index, state])
                if x == COL - 1 and y == 0:
                    break
                x = x_next
                y = y_next
                action = max_action
            rewards[i] += rewards_sum
    rewards /= runs
    # avg_rewards = []
    # for i in range(9):
    #     avg_rewards.append(np.mean(rewards[:i+1]))
    # for i in range(10,len(rewards)+1):
    #     avg_rewards.append(np.mean(rewards[i-10:i]))
    return rewards

def showPlot(q_rewards, s_rewards):
    plt.plot(q_rewards, label="Q_learning")
    plt.plot(s_rewards, label="Sarsa")
    plt.legend(loc="lower right")
    plt.ylim(-100, 0)
    plt.show()

def showPath(q):
    for y in range(ROW - 1, -1, -1):
        for x in range(COL):
            state = get_state(x, y)
            action = action_list[np.argmax(q[:, state])]
            if action == 'up':
                print(" U ", end="")
            if action == 'down':
                print(" D ", end="")
            if action == 'left':
                print(" L ", end="")
            if action == 'right':
                print(" R ", end="")
        print()

def PrintPolicy(Q_s, Q_q):
    print("------------Sarsa Policy------------")
    showPath(Q_s)
    print("----------QLearning Policy----------")
    showPath(Q_q)
    print("------------------------------------")

#==========================================#
# n-step Sarsa and Sarsa-lambda, GAMMA=0.8 #
#==========================================#

GAMMA_1 = 0.8

def n_step_Sarsa(q, N):
    counts = np.zeros([150])
    for i in range(150):
        x = 0
        y = 0
        action = epsilon_greedy(x, y, q, EPSILON)
        T = np.Infinity
        t = 0
        count = 0
        reward_list, used_action, state_list = [0], [], []
        used_action.append(action)
        state_list.append(get_state(x, y))
        while True:
            count += 1
            if t < T:
                x_next, y_next, reward = move_reward(x, y, action)
                reward_list.append(reward)
                state_list.append(get_state(x_next, y_next))
                if x_next == COL - 1 and y_next == 0:
                    x_next = 0
                    y_next = 0
                    T = t + 1
                else:
                    next_action = epsilon_greedy(x_next, y_next, q, EPSILON)
                    used_action.append(next_action)
            tau = t - N + 1
            if tau >= 0:
                G = 0
                for k in range(tau+1, min(tau+N, T) + 1):
                    G += GAMMA**(i-tau-1)*reward_list[k]
                if tau + N < T:
                    s, a = state_list[tau+N], action_list.index(used_action[tau+N])
                    G += GAMMA**N * q[a, s]
                s, a = state_list[tau], action_list.index(used_action[tau])
                q[a, s] += ALPHA * (G - q[a, s])
            if tau == T - 1:
                break
            t += 1
            action = next_action
            x = x_next
            y = y_next
        counts[i] += count
    return counts

def Sarsa_lambda(q, lam):
    counts = np.zeros([150])
    for i in range(150):
        count = 0
        Z = np.zeros((4, COL * ROW))
        x, y = 0, 0
        action = action_list[0]
        while True:
            count += 1
            action_index = action_list.index(action)
            x_next, y_next, reward = move_reward(x, y, action)
            next_action = epsilon_greedy(x_next, y_next, q, EPSILON)
            next_action_index = action_list.index(next_action)
            TD_error = reward + GAMMA_1 * q[next_action_index, get_state(x_next, y_next)] - q[action_index, get_state(x, y)]
            Z[action_index, get_state(x, y)] += 1
            for a in range(4):
                for s in range(ROW * COL):
                    q[a, s] += ALPHA * TD_error * Z[a, s]
                    Z[a, s] *= GAMMA_1 * lam
            action = next_action
            x = x_next
            y = y_next
            if x_next == COL - 1 and y_next == 0:
                break
        counts[i] += count
    return counts

def main():
    #=======================================#
    # Q learning and Sarsa on cliff walking #
    #=======================================#

    q_gird = np.zeros((4, COL * ROW))
    q_rewards = Q_learning(q_gird)
    s_gird = np.zeros((4, COL * ROW))
    s_rewards= Sarsa(s_gird)
    showPlot(q_rewards, s_rewards)
    PrintPolicy(s_gird, q_gird)

    #================================#
    # n-step Sarsa and Sarsa(lambda) #
    #================================#
    print("--------n-step Sarsa Policy---------")
    print()
    for n in [1, 3, 5]:
        n_gird = np.zeros((4, COL * ROW))
        n_step = n_step_Sarsa(n_gird, n)
        print("---------------n = %s----------------" %str(n))
        showPath(n_gird)
        plt.plot(n_step, label="n = " + str(n))
    print("------------------------------------")
    plt.legend()
    plt.show()

    print("-------Sarsa(lambda) Policy---------")
    print()
    for lam in [0, 0.5, 1]:
        l_gird = np.zeros((4, COL * ROW))
        l_step = Sarsa_lambda(l_gird, lam)
        print("-----------lambda = %s---------------" %str(lam))
        showPath(l_gird)
        plt.plot(l_step, label="lambda = " + str(lam))
    print("------------------------------------")
    plt.legend()
    plt.ylim(0, 400)
    plt.show()

if __name__ == "__main__":
    main()
    