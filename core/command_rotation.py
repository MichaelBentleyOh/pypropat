import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from kinematics import Kinematics  # quaternion → 회전행렬 변환 함수가 있어야 함
import sys

# 터미널 입력을 받아 Queue에 전달하는 함수 (Windows에서는 CONIN$를 다시 열어줍니다)
def input_process(input_queue):
    try:
        # Windows에서 자식 프로세스가 콘솔 입력을 받도록 재개방
        sys.stdin = open('CONIN$', 'r')
    except Exception as e:
        print("stdin 재개방 실패:", e)
    while True:
        try:
            user_input = input("새로운 Euler 각 입력 (yaw pitch roll, 단위: deg): ")
        except EOFError:
            print("EOFError 발생!")
            break
        try:
            # 입력받은 문자열을 실수 4개로 변환
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

# 본체 좌표계(화살표)를 그려주는 함수
def draw_body_axes(ax, rotation_matrix):
    b1 = rotation_matrix @ np.array([2, 0, 0])
    b2 = rotation_matrix @ np.array([0, 2, 0])
    b3 = rotation_matrix @ np.array([0, 0, 2])
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
    # 입력 받을 Queue 생성
    input_queue = multiprocessing.Queue()

    # 입력 프로세스 시작 (daemon=True를 사용하면 메인 종료시 같이 종료됨)
    proc = multiprocessing.Process(target=input_process, args=(input_queue,), daemon=True)
    proc.start()

    # 초기 quaternion (w, x, y, z) - 단위 quaternion
    control_quat = np.array([1, 0, 0, 0])

    # Matplotlib 인터랙티브 모드 활성화
    plt.ion()
    fig = plt.figure("터미널 입력에 따른 큐브 회전 (multiprocessing)")
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])
    ax.view_init(elev=20, azim=27)

    # 큐브의 정점 정의 및 면(face) 구성
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

    print("프로그램 시작. 터미널에 quaternion 값을 입력하세요 (예: '0.9239 0 0.3827 0').")

    try:
        while True:
            # Queue에 새 입력이 있다면 control_quat 갱신
            if not input_queue.empty():
                new_quat = input_queue.get()
                control_quat = new_quat
                print("새 quaternion 적용:", control_quat)

            # control_quat를 이용하여 회전행렬 계산 후 큐브 회전 적용
            rotation_matrix = Kinematics.quatrmx(control_quat)
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
            draw_body_axes(ax, rotation_matrix)

            fig.canvas.draw_idle()
            plt.pause(0.01)  # 10ms 간격 업데이트
    except KeyboardInterrupt:
        print("프로그램 종료합니다.")

if __name__ == '__main__':
    multiprocessing.freeze_support()  # Windows용: exe 생성 시 필요할 수 있음
    main()
