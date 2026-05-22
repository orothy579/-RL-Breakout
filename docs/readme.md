Hello everyone. It is time to begin our Final Project, where you will apply the reinforcement learning theories we have studied throughout the semester to design and train an optimal agent.

The target environment for this project is ALE/Breakout-v5. The core objective of this project is not merely to achieve a high score, but to demonstrate a deep understanding of RL principles through rigorous experimentation and analysis.

1. Deadline and SubmissionDeadline:
Wednesday, June 10, 2026, at 24:00 (Midnight)
Report Format: Free format. Ensure your report logically presents your experiments, results, and critical reflections.

2. Evaluation Criteria (Crucial)
This project is heavily weighted toward your process and analysis rather than the raw final score.
Depth of Experimentation: We expect to see a variety of attempts, such as tuning hyperparameters, comparing different algorithms (e.g., DQN vs. PPO), or analyzing the effects of different preprocessing methods (e.g., Frame Stacking). 
Meaningful Reflection: You must interpret your experimental data based on reinforcement learning theory. Explain the "why" behind the results you observed.
Penalty Factors: Simply implementing an external library (like Stable Baselines3 or CleanRL) and running it without any theoretical reflection or original analysis will not result in a good grade. You must justify your choice of library, architecture, and hyperparameters. 

3. Mandatory Environment Setup (Baseline)
To ensure evaluation fairness, all students must conduct and report their official results using the following fixed baseline configuration: 

Required Packages: gymnasium[atari]==1.3.0, ale-py==0.11.2, autorom[accept-rom-license]==0.6.1.

4. Attachment Guide
Installation Manual: Comprehensive guide for Anaconda virtual environment setup and package installation.  
Guide to the Breakout Environment: Detailed explanation of environment parameters and troubleshooting for common errors like dimensional permutation (HWC to CHW).  
requirements.txt: File for installing the common baseline libraries.  
verify_env.ipynb: A verification script to immediately check if your environment is correctly set up.  

I hope this project serves as an opportunity to bridge the gap between RL theory and practice while gaining your own unique insights. I look forward to seeing your in-depth research results!