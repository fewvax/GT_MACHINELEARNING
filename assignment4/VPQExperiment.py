import gymnasium as gym
from hiive.mdptoolbox.mdp import ValueIteration, PolicyIteration, QLearning
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


class Experiment():
    def __init__(self, prob, reward) -> None:
        self.prob = prob
        self.reward = reward
        self.vi_results = []
        self.pi_results = []
        self.qlearn_results1 = []
        self.qlearn_results2 = []
        self.qlearn_results3 = []
        self.df_vi_results = pd.DataFrame()
        self.df_pi_results = pd.DataFrame()
        self.df_qlearn_results1 = pd.DataFrame()
        self.df_qlearn_results2 = pd.DataFrame()
        self.df_qlearn_results3 = pd.DataFrame()
        self.max_it = 1e9
        self.gamma = [0.1, 0.3, 0.5, 0.8, 0.9]
        self.eps = [0.1, 0.3, 0.5, 0.7, 1]
        self.alpha = [0.1, 0.3, 0.5, 0.7, 1]
        self.val()
        self.pol()
        self.ql()


    def val(self):
        for g in self.gamma:
            vi = ValueIteration(self.prob, self.reward, gamma=g, max_iter=self.max_it)
            start = time.time()
            temp_df = pd.DataFrame(vi.run())
            print("vi" , g, time.time()-start)
            temp_df['gamma'] = g
            self.df_vi_results = self.df_vi_results.append(temp_df)
            self.vi_results.append(vi.policy)

    def pol(self):
        for g in self.gamma:
            pi = PolicyIteration(self.prob, self.reward, gamma=g, max_iter=self.max_it)
            start = time.time()
            temp_df = pd.DataFrame(pi.run())
            print("pi" , g, time.time()-start)
            temp_df['gamma'] = g
            self.df_pi_results = self.df_pi_results.append(temp_df)
            self.pi_results.append(pi.policy)
    
    def ql(self):
        for e in self.eps:
            for a in self.alpha:
                q = QLearning(self.prob, self.reward, gamma=0.1, epsilon=e, alpha=a, n_iter=(self.max_it/10000))
                start = time.time()
                temp_df = pd.DataFrame(q.run())
                print("q" , e, a, '0.1', time.time()-start)
                temp_df['start_epsilon'] = e
                temp_df['start_alpha'] = a
                self.df_qlearn_results1 = self.df_qlearn_results1.append(temp_df)
                self.qlearn_results1.append(q.policy)
                q = QLearning(self.prob, self.reward, gamma=0.5, epsilon=e, alpha=a, n_iter=(self.max_it/10000))
                start = time.time()
                temp_df = pd.DataFrame(q.run())
                print("q" , e, a, '0.5', time.time()-start)
                temp_df['start_epsilon'] = e
                temp_df['start_alpha'] = a
                self.df_qlearn_results2 = self.df_qlearn_results2.append(temp_df)
                self.qlearn_results2.append(q.policy)
                q = QLearning(self.prob, self.reward, gamma=0.9, epsilon=e, alpha=a, n_iter=(self.max_it/10000))
                start = time.time()
                temp_df = pd.DataFrame(q.run())
                print("q" , e, a, '1', time.time()-start)
                temp_df['start_epsilon'] = e
                temp_df['start_alpha'] = a
                self.df_qlearn_results3 = self.df_qlearn_results3.append(temp_df)
                self.qlearn_results3.append(q.policy)
        
    
        

    def plot_gamma(self, df, met= 'Mean V'):
        for g in self.gamma:
            print(range(1,len(df[df.gamma==g]['Iteration'])+1))
            plt.plot(range(1,len(df[df.gamma==g]['Iteration'])+1),df[df.gamma==g][met],  label="gamma=" +str(g))
        plt.xlabel("Iteration")
        plt.ylabel('Reward')
        plt.legend()


    def plot_comparative(self, met= 'Mean V'):
        plt.plot( range(1, len(self.df_vi_results['Iteration'])+1), self.df_vi_results[met], label="Policy Iteration")
        plt.plot( range(1, len(self.df_pi_results['Iteration'])+1), self.df_pi_results[met], label="Value Iteration")
        plt.plot( range(1, len(self.df_qlearn_results1['Iteration'])+1), self.df_qlearn_results1[met], label="Q Learning gamma = 0.1")
        plt.plot( range(1, len(self.df_qlearn_results2['Iteration'])+1),  self.df_qlearn_results2[met], label="Q Learning gamma = 0.5")
        plt.plot(range(1,len(self.df_qlearn_results3['Iteration'])+1),  self.df_qlearn_results3[met],  label="Q Learning gamma = 0.9")
        plt.xlabel("Iteration")
        plt.ylabel('Reward')
        plt.legend()

    def plot_eps(self, df, met= 'Mean V'):
        for e in self.eps:
            plt.plot(range(1, len(df[df.start_epsilon==e]['Iteration'])+1), df[df.start_epsilon==e][met],  label="epsilon="+str(e))
        plt.xlabel("Iteration")
        plt.ylabel('Reward')
        plt.legend()

    def plot_alpha(self, df, met= 'Mean V'):
        for a in self.alpha:
            plt.plot( range(1, len(df[df.start_alpha==a]['Iteration'])+1), df[df.start_alpha==a][met], label="alpha="+str(a))
        plt.xlabel("Iteration")
        plt.ylabel('Reward')
        plt.legend()

    