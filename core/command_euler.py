import sys
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from scipy.integrate import solve_ivp
from control import Control        # rigidbody_dynamics() 사용
from kinematics import Kinematics  # ezyxrmx, quatrmx, rmxquat 함수 사용

# 터미널 입력을 받아 Euler 각(°)을 quaternion으로 변환 후 Queue에 전달하는 프로세스 함수
def input_process(input_queue):
    try:
        sys.stdin = open("CONIN$", "r")
    except Exception as e:
        print("stdin 재개방 실패:", e)
    while True:
        try:
            user_input = input("새로운 Euler 각 입력 (yaw pitch roll, 단위: deg): ")
        except EOFError:
            print("EOFError 발생!")
            break
        try:
            arr = np.array([float(val) for val in user_input.split()])
            if len(arr) != 3:
                print("3개의 값을 입력하세요! (yaw, pitch, roll)")
                continue
            # 입력된 각도를 radian으로 변환
            angles_rad = np.deg2rad(arr)
            # Kinematics.ezyxrmx()를 이용해 회전행렬 계산 (Z-Y-X, 즉 yaw, pitch, roll)
            rotation_matrix = Kinematics.ezyxrmx(angles_rad)
            # Kinematics.rmxquat()를 통해 회전행렬 → quaternion 변환
            quat = Kinematics.rmxquat(rotation_matrix)
            input_queue.put(quat)
            print("새로운 quaternion (unit):", quat)
        except Exception as e:
            print("입력 오류:", e)

# 큐브 면(face) 업데이트 함수
def update_cube(cube_poly, rotation_matrix, cube_vertices):
    rotated_vertices = (rotation_matrix @ cube_vertices.T).T
    new_faces = [
        [rotated_vertices[0], rotated_vertices[1], rotated_vertices[2], rotated_vertices[3]],
        [rotated_vertices[4], rotated_vertices[5], rotated_vertices[6], rotated_vertices[7]],
        [rotated_vertices[0], rotated_vertices[1], rotated_vertices[5], rotated_vertices[4]],
        [rotated_vertices[2], rotated_vertices[3], rotated_vertices[7], rotated_vertices[6]],
        [rotated_vertices[1], rotated_vertices[2], rotated_vertices[6], rotated_vertices[5]],
        [rotated_vertices[4], rotated_vertices[7], rotated_vertices[3], rotated_vertices[0]]
    ]
    cube_poly.set_verts(new_faces)

# 본체 좌표계(화살표) 업데이트 함수
def draw_body_axes(ax, rotation_matrix):
    b1 = rotation_matrix @ np.array([2, 0, 0])
    b2 = rotation_matrix @ np.array([0, 2, 0])
    b3 = rotation_matrix @ np.array([0, 0, 2])
    # 기존에 그려진 축(Line3DCollection) 삭제
    for artist in ax.collections:
        if isinstance(artist, Line3DCollection):
            try:
                artist.remove()
            except Exception:
                pass
    ax.quiver(0, 0, 0, b1[0], b1[1], b1[2], color='r', linewidth=2)
    ax.quiver(0, 0, 0, b2[0], b2[1], b2[2], color='g', linewidth=2)
    ax.quiver(0, 0, 0, b3[0], b3[1], b3[2], color='b', linewidth=2)

def main():
    # 입력 프로세스 시작
    input_queue = multiprocessing.Queue()
    proc = multiprocessing.Process(target=input_process, args=(input_queue,), daemon=True)
    proc.start()

    deg2rad = np.pi / 180
    # 초기 상태: quaternion과 각속도
    initial_quat = np.array([np.cos(22.5 * deg2rad), 0, 0, np.sin(22.5 * deg2rad)])
    initial_omega = np.array([0, 0, 0])
    state = np.hstack((initial_quat, initial_omega))

    # 초기 제어 목표
    desired_quat = np.array([1, 0, 0, 0])
    desired_omega = np.zeros(3)
    desired_state = np.concatenate((desired_quat, desired_omega))
    print("초기 desired_state (제어 목표):", desired_state)

    ext_torque = np.zeros(3)         # 외부 토크 없음
    tensin = np.diag([10, 10, 10])     # 관성 텐서
    p_gain = 20
    d_gain = 80

    # Matplotlib 플롯 설정
    plt.ion()
    fig = plt.figure("Rigid Body Control")
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=20, azim=27)

    # 큐브 정점 및 면 설정
    cube_vertices = np.array([
        [1, 1, 1],
        [1, 1, -1],
        [1, -1, -1],
        [1, -1, 1],
        [-1, 1, 1],
        [-1, 1, -1],
        [-1, -1, -1],
        [-1, -1, 1]
    ])
    cube_faces = [
        [cube_vertices[0], cube_vertices[1], cube_vertices[2], cube_vertices[3]],
        [cube_vertices[4], cube_vertices[5], cube_vertices[6], cube_vertices[7]],
        [cube_vertices[0], cube_vertices[1], cube_vertices[5], cube_vertices[4]],
        [cube_vertices[2], cube_vertices[3], cube_vertices[7], cube_vertices[6]],
        [cube_vertices[1], cube_vertices[2], cube_vertices[6], cube_vertices[5]],
        [cube_vertices[4], cube_vertices[7], cube_vertices[3], cube_vertices[0]]
    ]
    cube_poly = Poly3DCollection(cube_faces, alpha=0.5, linewidths=1, edgecolors='k')
    ax.add_collection3d(cube_poly)

    print("프로그램 시작. 터미널에 Euler 각 값을 입력하면 제어 목표가 업데이트됩니다. 예: 10 20 30 (yaw, pitch, roll)")

    
    control_interval = 0.1
    t = 0.0
    while True:
        # 터미널 입력이 있다면 desired_state 업데이트
        if not input_queue.empty():
            new_desired_quat = input_queue.get()
            desired_quat = new_desired_quat
            desired_state = np.concatenate((desired_quat, np.zeros(3)))
            print("새로운 desired_state 적용:", desired_state)

        # 제어 목표는 고정된 채로 control_interval 동안 ODE 통합 수행
        sol = solve_ivp(
            lambda time, y: Control.rigidbody_dynamics(time, y, ext_torque, tensin,
                                                         p_gain=p_gain, d_gain=d_gain,
                                                         desired_state=desired_state),
            (t, t + control_interval), state, t_eval=[t + control_interval],
            method='RK45'
        )
        # 통합된 최종 상태를 업데이트
        state = sol.y[:, -1]
        t = sol.t[-1]

        # drift 방지를 위해 quaternion 정규화
        quat = state[0:4]
        quat = quat / np.linalg.norm(quat)
        state[0:4] = quat

        # 현재 quaternion으로부터 회전행렬 계산 및 시각적 업데이트
        rotation_matrix = Kinematics.quatrmx(quat)
        update_cube(cube_poly, rotation_matrix, cube_vertices)
        draw_body_axes(ax, rotation_matrix)

        fig.canvas.draw_idle()
        plt.pause(0.02)

if __name__ == '__main__':
    multiprocessing.freeze_support()  # Windows에서 필요할 수 있음
    main()
