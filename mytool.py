import os
import shutil
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def create_or_reset_directory(directory_path):
    """
    Checks if the specified directory exists.
    If it exists, it asks the user whether to delete it.
    If the user agrees, it deletes the directory and creates a new one.
    If the directory does not exist, it creates a new one.

    Args:
        directory_path (str): The path of the directory to create or reset.

    Returns:
        bool: True if a new directory was created or an existing one was reset,
              False if the existing directory was kept or an error occurred.
    """
    if os.path.exists(directory_path):
        response = input(f"Directory '{directory_path}' already exists. Do you want to delete it and create a new one? (y/n): ").lower()
        if response == 'y':
            try:
                shutil.rmtree(directory_path)
                os.makedirs(directory_path)
                print(f"Directory '{directory_path}' has been deleted and a new one created.")
                return True  # New directory created successfully
            except OSError as e:
                print(f"Error: Failed to delete directory '{directory_path}': {e}")
                return False # Failed to delete
        else:
            print(f"Directory '{directory_path}' will be kept as is.")
            return False # Existing directory kept
    else:
        try:
            os.makedirs(directory_path)
            print(f"Directory '{directory_path}' has been created.")
            return True  # New directory created successfully
        except OSError as e:
            print(f"Error: Failed to create directory '{directory_path}': {e}")
            return False # Failed to create


def plot_z_on_plane(csvfile):
    # CSVファイルを読み込む
    #df = pd.read_csv(csvfile, header=None, names=['index', 'x', 'y'], sep=' ')
    df = pd.read_csv(csvfile, header=None, names=['x', 'y'], sep=' ')

    # x, y座標のデータを取得
    x = df['x']
    y = df['y']
    
    # 矢印の始点と終点を設定
    start_x = x[:-1]
    start_y = y[:-1]
    end_x = x[1:]
    end_y = y[1:]
    
    # グラフを描画
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'o-')  # 点と線をプロット
    
    # 矢印を描画
    plt.quiver(start_x, start_y, end_x - start_x, end_y - start_y, angles='xy', scale_units='xy', scale=1)
    
    # グラフを表示
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Plot with Arrows')
    plt.show()




def save_shape(img, output_file):
    """
    img is a ndArray, which is typically an output by the decoder. The elelments of the array ranges from 0.0 to 1.0.
    """
    #img = np.full((self.nx, self.ny), 1) - img # Make the material and air as black and air, respectively
    img = 1.0 - img # Make the material and air as black and air, respectively
    img = img * 255
    img = img.astype(int)
    cv2.imwrite(output_file, img)


def create_directory(directory_path):
    """
    指定されたディレクトリが存在する場合、削除して新たに作成するかどうかをユーザーに尋ねる。

    Args:
        directory_path (str): 作成したいディレクトリのパス

    Returns:
        bool: ディレクトリが作成されたかどうか
    """

    if os.path.exists(directory_path):
        # ディレクトリが存在する場合
        answer = input(f"ディレクトリ {directory_path} は既に存在します。削除して新たに作成しますか？(yes/no): ")
        if answer.lower() == "yes":
            # ディレクトリを削除
            #os.rmdir(directory_path)
            shutil.rmtree(directory_path)
            # 新たに作成
            os.makedirs(directory_path)
            return True
        else:
            print("ディレクトリは作成されませんでした。")
            return False
    else:
        # ディレクトリが存在しない場合
        os.makedirs(directory_path)
        return True


# 使用例
if __name__ == "__main__":
    directory = "my_new_directory"
    create_directory(directory)
