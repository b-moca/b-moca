import os
import random

import numpy as np

import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

from bmoca.agent.custom.networks import CustomAgentBackbone

from bmoca.agent.custom import utils
from bmoca.agent.custom.rl.replay_buffer import MultiTaskReplayBufferConcat, Transition_Task


_WORK_PATH = os.environ['BMOCA_HOME']
    

class CustomRLCriticQ(CustomAgentBackbone):
    def __init__(self, 
                text_enc_name=f"{_WORK_PATH}/asset/agent/Auto-UI-Base",
                action_shape=[385]
                ):
        super().__init__(text_enc_name=text_enc_name)
        
        self.prediction_head = nn.Sequential(
            nn.Linear(self.text_config.hidden_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, action_shape[0]),
            nn.Sigmoid() # for scaling
        )
    
    def forward(self,\
                goal_input_ids,\
                goal_attention_mask,\
                img_observation,\
                eval_mode=False,
        ):
        """
        args:
            screen_input: processed screen image with keys of (flattened_patches, attention_mask)
            query_input: tokenized query text with keys of (input_ids, attention_mask)
        """
        query_input = {
            'input_ids': goal_input_ids,
            'attention_mask': goal_attention_mask,
        }

        # textual encoding
        with torch.no_grad():
            self.text_enc.eval()
            text_features = self.text_enc.encode(**query_input)

        # visual encoding 
        img_features = self.img_enc(img_observation)
        if len(img_features.size()) == 2:
            img_features = img_features.unsqueeze(1) # [B, hidden_dim] -> [B, 1, hidden_dim]

        # attention
        feat_att, _ = self.mha_layer(text_features, img_features, img_features)

        feat_att = torch.mean(feat_att, axis=1) # [B, S, E] -> [B, E]
        
        values = self.prediction_head(feat_att)
        return values

class DDQN:
    def __init__(
        self,
        action_shape = [4],
        td_step = 1,
        # modules
        feature_dim = 768,
        critic_hidden_dim = 512,
        #training
        device = "cuda",
        lr = 2e-4,
        critic_tau = 0.01,
        critic_gamma = 0.99,
        # stddev
        epsilon_schedule=None,
        # save
        checkpoint_name=None,
        buffer_size=10000,
        avail_tasks=None,
        succ_sample_ratio=0.25
    ):
        
        self.device = device
        self.succ_sample_ratio = succ_sample_ratio
        self.action_shape = action_shape
        self.td_step = td_step
        self.feature_dim = feature_dim
        self.critic_hidden_dim = critic_hidden_dim
        self.lr = lr
        self.critic_tau = critic_tau
        self.critic_gamma = critic_gamma
        self.epsilon_schedule = epsilon_schedule
        self.training = True
        self.avail_tasks = avail_tasks
        
        # checkpoint
        self.checkpoint_name = checkpoint_name       
         
        # modules
        self.network_ids = ['critic1', 'critic_target1', 'critic2', 'critic_target2']
        
        self.critic1 = CustomRLCriticQ(action_shape=action_shape).to(device)
        self.critic_target1 = CustomRLCriticQ(action_shape=action_shape).to(device)
        
        self.critic2 = CustomRLCriticQ(action_shape=action_shape).to(device)
        self.critic_target2 = CustomRLCriticQ(action_shape=action_shape).to(device)
      
        # init targets
        self.critic_target1.load_state_dict(self.critic1.state_dict())
        self.critic_target2.load_state_dict(self.critic2.state_dict())
                
        # optimizers
        self.critic_optimizer1 = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic_optimizer2 = optim.Adam(self.critic2.parameters(), lr=lr)

        # replay buffer
        self.succ_replay_buffer = MultiTaskReplayBufferConcat(capacity=buffer_size, task_instructions=self.avail_tasks)
        self.fail_replay_buffer = MultiTaskReplayBufferConcat(capacity=buffer_size, task_instructions=self.avail_tasks)
        self.global_step = 0
        self.training = True
        self.set_train()
        
        
    def set_train(self):
        self.critic1.train()
        self.critic_target1.train()
        self.critic2.train()
        self.critic_target2.train()
        
    def set_eval(self):
        self.critic1.eval() 
        self.critic_target1.eval()
        self.critic2.eval()
        self.critic_target2.eval()
        
    def save(self, filename="./tmp.pt"):
        save_path = os.path.join(f"{_WORK_PATH}/results/{self.checkpoint_name}", filename)
        save_dict = {}
        for id in self.network_ids:
            save_dict[id] = eval(f"self.{id}.state_dict()")
        save_dict["critic_optimizer1"] = self.critic_optimizer1.state_dict()
        save_dict["critic_optimizer2"] = self.critic_optimizer2.state_dict()
        torch.save(save_dict, save_path)
        return
    
    def load(self, filename="./tmp.pt"):
        checkpoint = torch.load(filename)
        self.critic1.load_state_dict(checkpoint["critic1"])
        self.critic_target1.load_state_dict(checkpoint["critic_target1"])
        self.critic2.load_state_dict(checkpoint["critic2"])
        self.critic_target2.load_state_dict(checkpoint["critic_target2"])            
        return
    
    def get_action_from_timesteps(self, timestep):
        self.set_eval()
        obs = np.stack([timestep.curr_obs['pixel']]) 
        obs = torch.from_numpy(obs).float().to(self.device)
        obs = obs.permute(0, 3, 1, 2)        
       
        input_ids, attention_mask = utils.tokenize_instruction(timestep.instruction)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        with torch.no_grad():
            q_value1 = self.critic1(img_observation=obs, 
                                    goal_input_ids=input_ids, 
                                    goal_attention_mask=attention_mask).cpu()
            q_value2 = self.critic2(img_observation=obs, 
                                    goal_input_ids=input_ids, 
                                    goal_attention_mask=attention_mask).cpu()
            
            q_value = (q_value1 + q_value2) / 2.0
            actions = q_value.argmax(dim=1).numpy()

        if self.training:
            epsilon = utils.schedule(f"{self.epsilon_schedule}", self.global_step)
            randoms = (np.random.random(size=(1,)) < epsilon)
            for i in range(len(actions)):
                actions[i] = random.randint(0, self.action_shape[0]-1) if randoms[i] else actions[i]
        self.set_train()
        return actions
    
    
    def episode_to_buffer(self, episode):
        reward_weight = self.critic_gamma ** np.arange(self.td_step)
        
        ep_rew = episode["reward"]
        ep_rew_sum = np.sum(ep_rew * (self.critic_gamma ** np.arange(len(ep_rew))))
        
        for t in range(episode["length"]):
            state = episode["observation"][t]
            action = episode["action"][t]            
            n_rewards_sum = np.sum(episode["reward"][t:t+self.td_step] * reward_weight)
            n_next_state = episode["next_observation"][t]
            done = episode["done"][t]
            task = episode["instruction"]
            if ep_rew_sum > 0:
                self.succ_replay_buffer.push(state, action, n_rewards_sum, n_next_state, done, task)
            else:
                self.fail_replay_buffer.push(state, action, n_rewards_sum, n_next_state, done, task)
            
            if done: break
            
        del episode
        return
    
    def update_critic(self,
                      state_batch,
                      action_batch, 
                      reward_batch,
                      next_state_batch,
                      done_batch,
                      input_ids_batch,
                      attention_mask_batch,
                      step):
        
        # Compute the target Q value
        with torch.no_grad():
            Q_value1 = self.critic_target1(img_observation=next_state_batch, 
                                           goal_input_ids=input_ids_batch, 
                                            goal_attention_mask=attention_mask_batch)
            Q_value2 = self.critic_target2(img_observation=next_state_batch, 
                                           goal_input_ids=input_ids_batch, 
                                            goal_attention_mask=attention_mask_batch)
            
            max_indices1 = torch.argmax(Q_value1, dim=1, keepdim=True)
            max_indices2 = torch.argmax(Q_value2, dim=1, keepdim=True)
            
            Q_target1 = Q_value1.gather(1, max_indices2)
            Q_target2 = Q_value2.gather(1, max_indices1)
            
            target_Q_value1 = (reward_batch.unsqueeze(1) + (self.critic_gamma**self.td_step * (1 - done_batch.unsqueeze(1)) * Q_target1).detach()).float()
            target_Q_value2 = (reward_batch.unsqueeze(1) + (self.critic_gamma**self.td_step * (1 - done_batch.unsqueeze(1)) * Q_target2).detach()).float()


        # Update the critic
        current_Q1 = self.critic1(img_observation=state_batch, 
                                  goal_input_ids=input_ids_batch, 
                                  goal_attention_mask=attention_mask_batch)
        current_Q2 = self.critic2(img_observation=state_batch, 
                                  goal_input_ids=input_ids_batch, 
                                  goal_attention_mask=attention_mask_batch)
                
        current_Q1 = current_Q1.gather(1, action_batch)
        current_Q2 = current_Q2.gather(1, action_batch)

        critic_loss1 = F.mse_loss(current_Q1, target_Q_value1)

        
        self.critic_optimizer1.zero_grad(set_to_none=True)
        critic_loss1.backward()
        self.critic_optimizer1.step()
        
        critic_loss2 = F.mse_loss(current_Q2, target_Q_value2)
        self.critic_optimizer2.zero_grad(set_to_none=True)
        critic_loss2.backward()
        self.critic_optimizer2.step()
    
    
    def update(self, batch_size):
        self.set_train()
        
        succ_batch_size = int(batch_size * self.succ_sample_ratio)
        if len(self.succ_replay_buffer) < succ_batch_size:
            succ_batch_size = len(self.succ_replay_buffer)
        fail_batch_size = batch_size - succ_batch_size
        
        succ_transitions = self.succ_replay_buffer.sample(succ_batch_size)
        if succ_batch_size != 0:
            succ_batch = Transition_Task(*zip(*succ_transitions))
        fail_transitions = self.fail_replay_buffer.sample(fail_batch_size)
        fail_batch = Transition_Task(*zip(*fail_transitions))
        
        if succ_batch_size != 0:
            batch = Transition_Task(succ_batch.state + fail_batch.state,
                               succ_batch.action + fail_batch.action,
                               succ_batch.n_rewards_sum + fail_batch.n_rewards_sum,
                               succ_batch.n_next_state + fail_batch.n_next_state,
                               succ_batch.done + fail_batch.done,
                               succ_batch.task + fail_batch.task)
        else:
            batch = fail_batch
        
        new_action_batch = []
        for a in batch.action:
            if len(a.shape) == 2:
                new_action_batch.append(a[0])
            elif len(a.shape) == 1:
                new_action_batch.append(a)
            else:
                raise ValueError
            
        state_batch = torch.tensor(np.stack(batch.state), dtype=torch.float32).to(self.device).permute(0, 3, 1, 2)
        action_batch = torch.tensor(np.stack(new_action_batch), dtype=torch.int64).to(self.device)
        reward_batch = torch.tensor(np.stack(batch.n_rewards_sum), dtype=torch.float32).to(self.device)
        next_state_batch = torch.tensor(np.stack(batch.n_next_state), dtype=torch.float32).to(self.device).permute(0, 3, 1, 2)
        done_batch = torch.tensor(np.stack(batch.done), dtype=torch.float32).to(self.device)

        input_ids_batch, attention_mask_batch = utils.tokenize_instruction(batch.task)
        input_ids_batch = input_ids_batch.to(self.device)
        attention_mask_batch = attention_mask_batch.to(self.device)
                
        self.update_critic(state_batch, 
                           action_batch, 
                           reward_batch,
                           next_state_batch,
                           done_batch,
                           input_ids_batch=input_ids_batch,
                           attention_mask_batch=attention_mask_batch,
                           step = self.global_step
                           )
    
        utils.soft_update(self.critic1, self.critic_target1, self.critic_tau)
        utils.soft_update(self.critic2, self.critic_target2, self.critic_tau)
        self.global_step += 1
        return