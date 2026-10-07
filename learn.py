# ==========================================
# PHẦN 1: SETUP TỪ TÀI LIỆU (IMAGE)
# ==========================================
import numpy as np
import gymnasium as gym
import random
import time
from IPython.display import clear_output

# 1. Khởi tạo môi trường FrozenLake
# render_mode='ansi' để in ra text trên console
env = gym.make("FrozenLake-v1", render_mode='ansi')

# 2. Xây dựng Q-Table
# Lấy kích thước không gian hành động (4) và không gian trạng thái (16 ô)
action_space_size = env.action_space.n
state_space_size = env.observation_space.n

# Khởi tạo bảng Q với toàn bộ giá trị bằng 0
q_table = np.zeros((state_space_size, action_space_size))

# 3. Khởi tạo các tham số Q-Learning
num_episodes = 10000             # Tổng số ván chơi (tập) để huấn luyện
max_steps_per_episode = 100      # Số bước tối đa Agent được đi trong 1 ván

learning_rate = 0.1              # Tốc độ học (alpha)
discount_rate = 0.99             # Hệ số chiết khấu tương lai (gamma)

exploration_rate = 1.0           # Tỷ lệ đi bừa ban đầu (epsilon)
max_exploration_rate = 1.0       # Ngưỡng epsilon lớn nhất
min_exploration_rate = 0.01      # Ngưỡng epsilon nhỏ nhất (chốt chặn an toàn)
exploration_decay_rate = 0.001   # Tốc độ suy giảm epsilon sau mỗi ván

# ==========================================
# PHẦN 2: VÒNG LẶP HUẤN LUYỆN (TRAINING LOOP)
# Bổ sung để code hoàn chỉnh vòng lặp học tập
# ==========================================

rewards_all_episodes = []

for episode in range(num_episodes):
    # Reset môi trường về trạng thái S (Start) ở mỗi ván mới
    state, info = env.reset()
    done = False
    rewards_current_episode = 0
    
    for step in range(max_steps_per_episode):
        # Chọn hành động (Exploration vs Exploitation)
        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold > exploration_rate:
            # Khai thác (Exploitation): Chọn hành động có Q-value cao nhất
            action = np.argmax(q_table[state, :])
        else:
            # Khám phá (Exploration): Bốc bừa 1 hướng
            action = env.action_space.sample()
            
        # Tương tác với môi trường
        new_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Cập nhật Q-Table dựa trên phương trình Bellman
        # Q_mới = Q_cũ * (1 - alpha) + alpha * (R + gamma * max_Q_tương_lai)
        q_table[state, action] = q_table[state, action] * (1 - learning_rate) + \
            learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))
        
        state = new_state
        rewards_current_episode += reward
        
        if done:
            break
            
    # Áp dụng Epsilon Decay: Giảm dần tỷ lệ đi bừa theo cấp số nhân
    exploration_rate = min_exploration_rate + \
        (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)
        
    rewards_all_episodes.append(rewards_current_episode)

print("\nQuá trình huấn luyện hoàn tất!")
print("Bảng Q-Table cuối cùng (16 hàng - 4 cột):")
print(q_table)