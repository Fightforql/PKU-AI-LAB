from typing import List
import numpy as np
from utils import Particle

### 可以在这里写下一些你需要的变量和函数 ###
COLLISION_DISTANCE = 1
MAX_ERROR = 50000
kk=0.5
sigma_p = 0.09 
sigma_t = 0.09
### 可以在这里写下一些你需要的变量和函数 ###


def generate_uniform_particles(walls, N):
    """
    输入：
    walls: 维度为(xxx, 2)的np.array, 地图的墙壁信息，具体设定请看README关于地图的部分
    N: int, 采样点数量
    输出：
    particles: List[Particle], 返回在空地上均匀采样出的N个采样点的列表，每个点的权重都是1/N
    """
    all_particles: List[Particle] = []
    for _ in range(N):
        all_particles.append(Particle(1.0, 1.0, 1.0, 1.0/N))
    ### 你的代码 ###
    xmin = walls[:, 0].min()
    xmax = walls[:, 0].max()
    ymin = walls[:, 1].min()
    ymax = walls[:, 1].max()
    for i in range(N):
        all_particles[i].position[0] = np.random.uniform(xmin, xmax)
        all_particles[i].position[1] = np.random.uniform(ymin,ymax)
        all_particles[i].theta = np.random.uniform(-np.pi, np.pi)
        test = [int(all_particles[i].position[0]+0.5),int(all_particles[i].position[1]+0.5)]
        if (test in walls):
            i -= 1
    return all_particles


def calculate_particle_weight(estimated, gt):
    """
    输入：
    estimated: np.array, 该采样点的距离传感器数据
    gt: np.array, Pacman实际位置的距离传感器数据
    输出：
    weight, float, 该采样点的权重
    """
    weight = 1.0
    ### 你的代码 ###
    l2_distance = np.linalg.norm(estimated - gt,ord=2)
    weight = np.exp(-kk * l2_distance)
    ### 你的代码 ###
    return weight


def resample_particles(walls, particles: List[Particle]):
    """
    输入：
    walls: 维度为(xxx, 2)的np.array, 地图的墙壁信息，具体设定请看README关于地图的部分
    particles: List[Particle], 上一次采样得到的粒子，注意是按权重从大到小排列的
    输出：
    particles: List[Particle], 返回重采样后的N个采样点的列表
    """
    resampled_particles: List[Particle] = []
    N = len(particles)
    cumulative_weights = np.cumsum([p.weight for p in particles])
    for i in range(N):
        u = np.random.uniform(0, 1)
        for i in range(N):
            if u <= cumulative_weights[i]:
                selected_particle = particles[i]
                resampled_particles.append(Particle(selected_particle.position[0], selected_particle.position[1], selected_particle.theta, 1.0 / N))
                break
    for i in range(len(resampled_particles)):
        resampled_particles[i].position[0] += np.random.normal(0, sigma_p)
        resampled_particles[i].position[1] += np.random.normal(0, sigma_p)
        resampled_particles[i].theta += np.random.normal(0, sigma_t)
        resampled_particles[i].theta = (resampled_particles[i].theta+np.pi) % (np.pi*2)-np.pi
    return resampled_particles

def apply_state_transition(p: Particle, traveled_distance, dtheta):
    """
    输入：
    p: 采样的粒子
    traveled_distance, dtheta: ground truth的Pacman这一步相对于上一步运动方向改变了dtheta，并移动了traveled_distance的距离
    particle: 按照相同方式进行移动后的粒子
    """
    ### 你的代码 ###
    p.theta+=dtheta
    p.theta = (p.theta+np.pi) % (2*np.pi)-np.pi
    dx = traveled_distance*np.cos(p.theta)
    dy = traveled_distance*np.sin(p.theta)
    p.position[0] += dx
    p.position[1] += dy
    ### 你的代码 ###
    return p

def get_estimate_result(particles: List[Particle]):
    """
    输入：
    particles: List[Particle], 全部采样粒子
    输出：
    final_result: Particle, 最终的猜测结果
    """
    final_result = Particle()
    ### 你的代码 ###
    particles_sorted = sorted(particles, key=lambda p: p.weight, reverse=True)
    final_result = particles_sorted[0]
    ### 你的代码 ###
    return final_result