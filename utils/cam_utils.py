import cv2
import numpy as np
import glob

def __drawCorners(img, corners, save_path):
    """
    img: 原始图像
    corners: 角点
    """
    # 绘制角点
    for corner in corners:
        cv2.circle(img, tuple(map(int, corner[0])), 40, (0, 0, 255), -1)
    # 保存绘制角点的图像
    cv2.imwrite(save_path, img)

def __generate(chessboard_size, square_size):
    """
    生成棋盘格的3D点
    chessboard_size: 棋盘格内角点个数
    square_size: 每个方格的大小，单位为毫米
    """
    # 准备棋盘格的3D点，单位为毫米
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size

    return objp

if __name__ == '__main__':
    # 设置棋盘格内角点个数
    chessboard_size = (7, 7)
    # 每个方格的大小，单位为毫米
    square_size = 25
    # 图像文件夹路径
    imgs_path = r"../data/cam_calibration/origin"

    # 生成棋盘格的3D点
    objp = __generate(chessboard_size, square_size)

    # 读取棋盘格图像
    images = glob.glob(rf'{imgs_path}/*.jpg')  # 替换为实际图像路径

    # 用于存储所有图像的3D点和2D点
    objpoints = []  # 3D点
    imgpoints = []  # 2D点

    for img_file in images:
        img = cv2.imread(img_file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 查找棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
            # 保存绘制角点的图像
            # __drawCorners(img, corners, img_file.replace('origin', 'result'))
        else:
            print(f'未找到角点：{img_file}')

    # 执行相机标定
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # 打印标定结果
    print(f"投影误差：{ret}\n")
    print(f"相机矩阵：{camera_matrix}\n")
    print(f"畸变系数：{dist_coeffs}\n")