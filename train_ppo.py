import torch
from torch.utils.data import BatchSampler, SubsetRandomSampler
import os
import pygame
from tqdm import tqdm
import sys
from game.flappy_bird import FlappyBird
from ppo.models import ActionModel, ValueModel

image_size = 84

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

action_model = ActionModel().to(device)
value_model = ValueModel().to(device)


def play(rand=True):
    action_model.eval()
    data = []
    reward_sum = 0
    game = FlappyBird()

    state, reward, terminal = game.step(0)

    while 1:
        prob = action_model(state[None].to(device))[0].exp()
        if rand:
            action_dist = torch.distributions.Categorical(prob)
            action = action_dist.sample().item()
        else:
            action = prob.argmax().item()

        next_state, reward, terminal = game.step(action, last_state=state)
        data.append((state, action, reward, next_state, terminal))
        reward_sum += reward

        state = next_state
        if terminal:
            break
    return data, reward_sum
    

def compute_advantage(td_delta):
    advatage_list = []
    advatage = 0.0
    for delta in td_delta.squeeze(-1).tolist()[::-1]:
        advatage = 0.98 * 0.95 * advatage + delta
        advatage_list.append(advatage)
    advatage_list.reverse()
    advatage_tensor = torch.tensor(advatage_list).float().to(device)
    return advatage_tensor


optim_action = torch.optim.Adam(action_model.parameters(), lr=1e-5)
optim_value = torch.optim.Adam(value_model.parameters(), lr=1e-5)
buffer_size = 2048
num_iters = 200000
batch_size = 64
epochs = 10

os.makedirs('ppo/models', exist_ok=True)

action_model.train()
value_model.train()

os.environ["SDL_VIDEODRIVER"] = "dummy"
game = FlappyBird()
s, r, t = game.step(0)
data_buffer = []

best_score = 0.0

for iter in tqdm(range(num_iters)):
    t = False
    iter_reward = 0.0
    while not t:
        action_model.eval()
        prob = action_model(s[None].to(device))[0].exp()
        action_dist = torch.distributions.Categorical(prob)
        a = action_dist.sample().item()
        ns, r, t = game.step(a, last_state=s)
        data_buffer.append((s, a, r, ns, t))
        s = ns
        iter_reward += r
        
        if len(data_buffer) >= buffer_size:
            states, actions, rewards, next_states, terminals = zip(*data_buffer)
            states = torch.stack(states).float().to(device)
            actions = torch.tensor(actions).view(-1, 1).long().to(device)
            rewards = torch.tensor(rewards).view(-1, 1).float().to(device)
            next_states = torch.stack(next_states).float().to(device)
            terminals = torch.tensor(terminals).view(-1, 1).long().to(device)
            with torch.no_grad():
                tgt = rewards + (1 - terminals) * 0.98 * value_model(next_states)
                td_delta = tgt - value_model(states)
                advantage = compute_advantage(td_delta).view(-1, 1)
                log_probs_old = action_model(states).gather(dim=1, index=actions).detach()
            for epoch in range(epochs):
                action_model.train()
                for idx in BatchSampler(SubsetRandomSampler(range(len(states))), batch_size, drop_last=False):
                    log_prob_new = action_model(states[idx]).gather(dim=1, index=actions[idx])
                    ratio = (log_prob_new - log_probs_old[idx]).exp()
                    action_loss = -torch.min(ratio * advantage[idx], torch.clamp(ratio, 0.8, 1.2) * advantage[idx]).mean()
                    value_loss = torch.nn.functional.mse_loss(value_model(states[idx]), tgt[idx].detach())
                    optim_action.zero_grad()
                    optim_value.zero_grad()
                    action_loss.backward()
                    value_loss.backward()
                    optim_action.step()
                    optim_value.step()
            data_buffer.clear()
    s, r, t = game.step(0)
    if (iter + 1) % 10 == 0:
        test_score = sum([play(rand=False)[-1] for _ in range(10)]) / 10
        if test_score > best_score:
            best_score = test_score
            torch.save(action_model.state_dict(), 'ppo/models/best.pth')
            print(f'Saved new best model with score: {best_score}')
        print(f'Iter {iter + 1}, Episode reward: {iter_reward}, Testing Score: {test_score}')
    if (iter + 1) % 1000 == 0:
        torch.save(action_model.state_dict(), f'ppo/models/model_{iter + 1}.pth')
        print(f'Saved model at iter {iter + 1}')


pygame.quit()
sys.exit()