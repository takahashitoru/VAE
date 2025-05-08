import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
import nlopt
import math
import pandas as pd

import myvae
from solve_eigen import EigenvalueSolver
import myutil
from img2msh_donut import img2msh_donut

# Dimension of the latent space

z_dim = 2

# Path to the VAE model

#dir_ptfile = "save_caribou_bs200_ep200_lr-3"
#dir_ptfile = "save_nessie_bs200_ep200_lr-3"
dir_ptfile = "save_cat_bs200_ep200_lr-3"

# Set bounds

"""
# caribou
lb = [-2.0, -1.0]
ub = [2.0, 3.0]
"""
"""
# nessie
lb = [-3.0, -3.0]
ub = [5.0, 2.0]
"""
# cat
lb = [-5.0, -4.0]
ub = [4.0, 3.0]

"""
lb = [-3.0, -3.0]
ub = [3.0, 3.0]
"""
"""
lb = [-3.0, 0.0]
ub = [3.0, 3.0]
"""
"""
lb = [-3.0, -3.0]
ub = [3.0, 0.0]
"""
"""
lb = [] # [-float('inf'), -float('inf')]
ub = [] # [float('inf'), float('inf')]
"""

# 初期推定値

#image_indexs = [11] # caribou
#image_indexs = [12, 13] # nessie1, nessei2
image_indexs = [17, 18] # cat1, cat2

z_mean = []
for idx_image in image_indexs:
    ztmp = np.zeros(z_dim)
    with open(os.path.join(dir_ptfile, f"OUTPUT_z_stats_{idx_image}.txt"), 'r') as f:
        first_line = f.readline().strip()
        data = first_line.split()
        if len(data) >= z_dim:
            for i in range(z_dim):
                ztmp[i] = float(data[i])
            z_mean.append(ztmp)
        else:
            print("Error: Invalid number of data in the input file.")


zidxs = [0] # caribou, nessie1, cat1
x0 = z_mean[zidxs[0]]

"""
zidxs = [1] # nessie2, cat2
x0 = z_mean[zidxs[0]]
"""

"""
zidxs = [0, 1] # nessie1 & nessie2, cat1 & cat2
x0 = np.zeros(z_dim)
for zidx in zidxs:
    x0 += z_mean[zidx]
x0 = x0 / len(zidxs) # center of the means
"""

#objtype = 1
objtype = 2
#objtype = 3

if objtype == 1:
    # 目的とする固有値[mode(>=0), value]
    #targets = [[0, 10]]
    #targets = [[0, 100]]
    #targets = [[1, 100]]
    #targets = [[1, 30]]
    #targets = [[0, 50], [1, 100]]
    #targets = [[0, 10], [1, 20], [2, 30], [3, 90]]
    #targets = [[0, 10], [1, 30], [2, 50], [3, 90]]
    #targets = [[0, 10], [1, 30], [2, 40], [3, 90]]
    #targets = [[0, 10], [1, 30], [2, 50], [3, 80]]
    targets = [[2, 50], [3, 80]] #OK
        
elif objtype == 2:
    # 目的とする固有値[mode(>=0), low, high]
    #targets = [[0, 0, 5]] # caribou NG
    #targets = [[0, 0, 10]] # caribou OK
    #targets = [[0, 15, 30]] # caribou OK

    #targets = [[1, 50, 30]] # caribou OK, but horn is separated
    #targets = [[1, 0, 30], [2, 50, 80]] # caribou OK
    #targets = [[0, 0, 30], [1, 50, 80]] # caribou OK

    #targets = [[0, 80, 0]] # nessie OK
    #targets = [[0, 80, 100]] # nessie OK

    #targets = [[0, 80, 0]] # cat
    #targets = [[0, 80, 100]] # cat
    #targets = [[0, 20, 30]] # cat
    targets = [[0, 180, 200]] # cat    

    #targets = [[0, 0, 20], [0, 40, 0]] # NG
    #targets = [[1, 0, 20], [2, 40, 0]]
    #targets = [[0, 0, 10], [1, 0, 20], [2, 40, 0]]
    #targets = [[1, 40, 20]]
    #targets = [[1, 0, 20]]
    #targets = [[0, 0, 20], [1, 40, 20]]
    #targets = [[0, 0, 10], [1, 40, 20]]
    #targets = [[0, 0, 20], [1, 0, 40]]
    #targets = [[0, 40, 20], [1, 40, 20], [2, 40, 20]]
    #targets = [[0, 0, 20], [1, 40, 60]]
    #targets = [[1, 30, 50], [2, 80, 100]] # N55, p28
    #targets = [[2, 60, 80], [3, 100, 120]]
    #targets = [[1, 0, 20], [2, 40, 60]]
    #targets = [[1, 60, 80], [2, 100, 120]]
    #targets = [[2, 30, 50], [3, 80, 100]]
    
    #targets = [[0, 0, 30]] # pigeon
    #targets = [[0, 90, 100]]
    
    #targets = [[1, 50, 60]] # data0910
    #targets = [[0, 50, 60]]
    #targets = [[0, 0, 20]]
    #targets = [[1, 80, 50]]
    #targets = [[0, 0, 50], [0, 0, 80]]
    #targets = [[1, 0, 100], [2, 0, 150]]
    #targets = [[0, 80, 50], [1, 80, 50], [2, 80, 50]]
    #targets = [[0, 0, 50], [1, 0, 50], [2, 80, 50]]
    #targets = [[0, 0, 40], [1, 0, 50], [2, 80, 50]]
    #targets = [[0, 0, 30], [1, 0, 50], [2, 80, 50]] # eigenvalue solver failed
    #targets = [[0, 0, 20], [1, 0, 50], [2, 80, 50]]
    #targets = [[0, 0, 50], [1, 0, 50], [2, 80, 1000]]
    #targets = [[3, 80, 50]]
    #targets = [[2, 0, 50], [3, 80, 50]]
    #targets = [[0, 10, 20], [1, 30, 40], [2, 30, 40], [3, 80, 50]]
    #targets = [[0, 10, 20], [1, 30, 50], [2, 30, 50], [3, 80, 1000]]
    #targets = [[3, 0, 50], [3, 80, 1000]]
    #targets = [[0, 10, 20], [1, 30, 50], [2, 30, 50], [3, 80, 100]]
    #targets = [[0, 10, 20], [1, 30, 40], [2, 30, 40], [3, 90, 100]]
    #targets = [[0, 10, 20], [1, 30, 50], [2, 30, 50], [3, 90, 100]]
    #targets = [[0, 10, 20], [1, 20, 30], [2, 30, 40], [3, 80, 150]]
    #targets = [[2, 30, 50], [3, 80, 100]] # OK
    #targets = [[3, 30, 50], [4, 80, 100]]

    #targets = [[0, 90, 100]] # pigeon
    #targets = [[0, 20, 40]] # pigeon

    #targets = [[0, 0, 50]] # cat1-cat2
    
    #targets = [[1, 40, 20]] # caribou
    #targets = [[1, 0, 20], [2, 40, 60]]
    #targets = [[0, 0, 20], [1, 40, 60]]
    #targets = [[0, 0, 20], [1, 50, 80]]
    #targets = [[1, 40, 20], [2, 40, 60]]
    #targets = [[0, 0, 5]]
    #targets = [[0, 15, 30]] 
    #targets = [[1, 0, 20]]

    #targets = [[0, 80, 0]] # nessie
    #targets = [[1, 150, 100]] # nessie
    #targets = [[0, 80, 100]] # nessie
    

elif objtype == 3:
    #targets = [[1, 40, 20]] # caribou; exclude omega[1] from [20,40] -> OK
    #targets = [[3, 100, 80]] # exclude omega[3] from [80,100] -> NG
    #targets = [[0, 0, 20], [3, 100, 80]]
    #targets = [[1, 40, 20], [2, 40, 80]]
    #targets = [[1, 0, 20], [2, 40, 60]]

    #targets = [[0, 0, 5]] # caribou; z_mean[0] -> OK
    #targets = [[0, 15, 30]] # caribou; z_mean[0] -> OK
    #targets = [[1, 40, 20]] # caribou; z_mean[0] -> NG
    #targets = [[0, 20, 30]] # caribou; z_mean[0] -> NG
    #targets = [[1, 40, 20], [2, 40, 60]] # caribou; z_mean[0]
    #targets = [[0, 0, 20], [1, 40, 60]] # caribou; z_mean[0]
    #targets = [[1, 40, 60]] # caribou; z_mean[0]
    #targets = [[0, 0, 10], [1, 40, 60]] # caribou; z_mean[0]
    #targets = [[0, 0, 20], [1, 50, 80]] # caribou; z_mean[0]
    
    #targets = [[0, 0, 30]] # pigeon
    #targets = [[0, 90, 100]] # pigeon

    #targets = [[0, 100, 120]] # pigeon
    #targets = [[0, 20, 40]] # pigeon
    #targets = [[0, 60, 80]] # pigeon
    #targets = [[0, 90, 100]] # pigeon

    targets = [[0, 0, 50]] # cat1-cat2
    
    #targets = [[0, 0, 50]] # cactus-snake-loupe
    #targets = [[0, 50, 0]] # cactus-snake-loupe
    #targets = [[0, 50, 0], [1, 100, 120]] # cactus-snake-loupe
    #targets = [[0, 50, 0], [1, 120, 150]] # cactus-snake-loupe
    #targets = [[3, 100, 50]] # cactus-snake-loupe
    
else:
    raise ValueError(f"Invalid objtype={objtype}.")
print("# targets=", targets)

# Use double precision
torch.set_default_dtype(torch.float64) # torch.float32 by default

### # Create the directories
### model_dir = os.path.basename(dir_ptfile)

#model_dir += "-lu"
model_dir = "lu"

for l, u in zip(lb, ub):
    model_dir += f"_{l:.1f}_{u:.1f}"

model_dir += "-x"
if len(z_mean) == 0:
    for val in x0:
        model_dir += f"_{val:.2f}"
else:
    for zidx in zidxs:
        model_dir += f"_{zidx}"
    
model_dir = os.path.join(dir_ptfile, model_dir)
print("model_dir=", model_dir)

opt_dir = f"obj{objtype}"
opt_dir += f"-targets"
for val in targets:
    opt_dir += f"_{val}"
opt_dir = os.path.join(model_dir, opt_dir)
    
print("opt_dir=", opt_dir)
myutil.delete_and_recreate_directory(opt_dir)


# Load the VAE model
state = torch.load(os.path.join(dir_ptfile, "model.pt"))
image_size = state['image_size']
z_dim = state['z_dim']
print("image_size=", image_size)
print("z_dim=", z_dim)

# 画像サイズに応じてVAEモデルを選択
if image_size == 64:
    modelA = myvae.VAE64(z_dim)
else:
    raise ValueError(f"Unregistered image_size={image_size}. Exit.")

modelA.load_state_dict(state['model_state_dict'])
print("loaded")

modelA.eval() #モデルを評価モードに設定し、Dropout や Batch Normalization などの挙動を評価用に切り替え


# Check
if len(lb) != z_dim or len(ub) != z_dim or len(x0) != z_dim:
    raise ValueError(f"Inconsitent lengths.")

# 呼び出し回数と f, x の値を保存するリスト
call_counts = []
f_values = []
x_values = []
omega_values = []
call_count = 0

threshold = 1.0


E = 2.0 * 10 ** 11  # ヤング率
nu = 0.3  # ポアソン比
lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))  # ラメ定数
mu = E / (2.0 * (1.0 + nu))  # せん断弾性率
rho = 7800.0  # 密度

sigma = 0.0
n_eig = 16

solver = EigenvalueSolver(mesh_filename="", lam=lam, mu=mu, rho=rho, sigma=sigma, n_eig=n_eig)

def myfunc(x, grad):
    # グローバル変数
    global call_count
    
    x_tensor = torch.tensor(x, requires_grad=True) # shape is (n,)

    png_path = os.path.join(opt_dir, f"{call_count:03d}.png")
    msh_path = png_path.replace('.png', '.msh')
    modes_path = png_path.replace('.png', '_modes.png')
    
    with torch.no_grad(): # 勾配計算を無効にする
        modelA.z2png(x_tensor, png_path)

    #success = img2msh(png_path, msh_path, max_area=1.0, smooth_kernel_size=7)
    #success = img2msh(png_path, msh_path, max_area=1.0, smooth_kernel_size=7, magnification=1.5)
    success = img2msh_donut(png_path, msh_path, max_area=1.0, smooth_kernel_size=7, magnification=1.5)
    if success == False:
        raise ValueError("Failed to create a mesh from the image.")
    
    solver.mesh_filename = msh_path
    results = solver.solve()
    
    if results:
        eigenvalues, eigenvectors = results
        print("Eigenvalues:\n", eigenvalues)
        
        """
        # 全てのモードを可視化
        fig = solver.visualize_all_modes(scale=0.5)
        if fig:
            plt.show()
        
        # 特定のモードのデータを書き出す (例: 1番目のモード)
        solver.write_mode_data(0)
        solver.write_mode_data(2, filename_base="mode_output")
        """

        omegas = []
        for i, eigenvalue in enumerate(eigenvalues):
            if eigenvalue > threshold:
                omega = np.sqrt(eigenvalue)
                omegas.append(omega)

        if objtype == 1:
            if len(omegas) < len(targets):
                raise ValueError(f"Failed to obtain {len(targets)} eigenvalues.")
            # MSE を計算
            """
            targets_tensor = torch.tensor(targets) # targets_tensor を 1次元配列として作成
            omegas_tensor = torch.tensor(omegas[:len(targets)])
            mse = torch.mean((omegas_tensor - targets_tensor) ** 2)
            f = mse.item()
            """
            target_values_tensor = torch.zeros(len(targets))
            omega_values_tensor = torch.zeros(len(targets))
            for i, target in enumerate(targets):
                target_values_tensor[i] = target[1]
                omega_values_tensor[i] = omegas[target[0]]
            mse = torch.mean((omega_values_tensor - target_values_tensor) ** 2)
            f = mse.item()
            
        elif objtype == 2:
            #omegas_tensor = torch.tensor(omegas[:len(target)])
            #f = (50.0 - omegas_tensor[0].item()) * (omegas_tensor[0].item() - 100.0)
            f = 0.0
            for target in targets:
                mode = target[0]
                if target[1] > target[2]:
                    sign = 1.0 # Exclude this mode from this band
                    low = target[2]
                    high = target[1]
                else:
                    sign = - 1.0 # Include this mode from this band
                    low = target[1]
                    high = target[2]
                    
                omega = omegas[mode]
                #f += sign * (low - omega) * (omega - high)
                f += sign * (low - omega) * (omega - high) / ((high - low) / 2.0)**2

        elif objtype == 3:

            def sigmoid(x):
                return 1 / (1 + np.exp(-x))


            f = 0.0
            for target in targets:
                mode = target[0]
                omega = omegas[mode]
                if target[1] > target[2]: # Exclude this mode from this band
                    low = target[2]
                    high = target[1]
                    f += sigmoid(omega - low) + sigmoid(high - omega)
                    #f += sigmoid(omega - low) + sigmoid(high - omega) - 1.0 # same
                else: # Include this mode from this band
                    low = target[1]
                    high = target[2]
                    f += sigmoid(low - omega) + sigmoid(omega - high)

        else:
            raise ValueError(f"Invalide objtype={objtype}.")

            
        call_counts.append(call_count)
        f_values.append(f)
        x_values.append(x.tolist())
        #omega_values.append(omegas_tensor.tolist())
        #print("iter=", call_count, "f=", f, "x=", x, "omega=", omegas_tensor.tolist())
        omegas_float = [float(omega) for omega in omegas]
        omega_values.append(omegas_float)
        print("iter=", call_count, "f=", f, "x=", x, "omegas=", omegas_float)
                
    else:

        raise ValueError("固有値問題の解決に失敗しました。メッシュファイルを確認してください。")

    
    if grad.size > 0:
        raise ValueError(f"Gradient cannot be computed.")

    # グローバル変数を更新
    call_count += 1
    
    return f


opt = nlopt.opt(nlopt.LN_COBYLA, z_dim)
opt.set_lower_bounds(lb)
opt.set_upper_bounds(ub)
opt.set_min_objective(myfunc)
#opt.add_inequality_constraint(lambda x,grad: myconstraint(x,grad,2,0), 1e-8)
#opt.add_inequality_constraint(lambda x,grad: myconstraint(x,grad,-1,1), 1e-8)
#opt.set_xtol_rel(1e-6)
#opt.set_ftol_rel(1e-3)
opt.set_ftol_abs(1e-3)
opt.set_maxeval(100)

x = opt.optimize(x0)
print("result message = ", myutil.nlopt_result_message(opt)) # 結果に対応するメッセージ

minf = opt.last_optimum_value()
print("optimum at ", x)
print("minimum value = ", minf)
try:
    optimum_step = f_values.index(minf)
    omegas_float = [float(omega) for omega in omega_values[optimum_step]]
    print(f"optimum_step={optimum_step} f={minf} x={x} omegas={omegas_float}")

except ValueError:
    print(f"Could not find the index for f={optimum_step}.")
    

# プロット
plt.plot(call_counts, f_values, marker='o')
plt.xlabel("Iteration step")
plt.ylabel("f (MSE)")
plt.title(f"targets={targets}, x0={x0}")
#plt.yscale('log') # y軸を対数スケールに設定
plt.grid(True) # グリッド線を表示
plt.savefig(os.path.join(opt_dir, "history.eps"), format="eps")
plt.savefig(os.path.join(opt_dir, "history.png"), format="png")
#plt.show()

# データをテキストファイルに保存
#myutil.save_to_space_separated_text(call_counts, f_values, x_values, opt_dir, filename="history.txt")

# DataFrame を作成
data = {"call_counts": call_counts, "f_values": f_values}

# x_values の各要素を列に展開
for i in range(z_dim):
    data[f"x_values_col{i}"] = [x[i] for x in x_values]
    
# omega_values の各要素を列に展開
max_omega_len = max(len(omega) for omega in omega_values) if omega_values else 0  # omega_values が空でないことを確認
for i in range(max_omega_len):
    data[f"omega_values_col{i}"] = [omega[i] if len(omega) > i else np.nan for omega in omega_values]
    
df = pd.DataFrame(data)
    
# スペース区切りのテキストファイルに保存
df.to_csv(os.path.join(opt_dir, "history.txt"), sep=' ', index=False, header=False)

# 指定した固有モードの変形形状とミーゼス応力の画像を生成
eigenvalues, eigenvectors = solver.solve()

max_target_mode = max([target[0] for target in targets]) # max_mode >= 0

j = 1 # 1-based index
for i, eigenvalue in enumerate(eigenvalues):
    #if eigenvalue > threshold and j <= len(targets):
    #if eigenvalue > threshold and j - 1 <= max_target_mode:
    if eigenvalue > threshold and j - 1 <= 6:
        omega = math.sqrt(eigenvalue)
        title = f"Mode {j}:, Eigenvalue: {omega:.1f}"
        scale=0
        solver.visualize_mode(i, scale=scale, filename=os.path.join(opt_dir, f"opt_mode{j:02d}.png"), title=title)
        scale=2000
        solver.visualize_mode(i, scale=scale, filename=os.path.join(opt_dir, f"opt_mode{j:02d}_{scale}.png"), title=title)

        # 特定のモードのアニメーションを生成 (例: 1番目のモード, v=0.2)
        v=3000
        solver.animate_mode(i, v=v, filename=os.path.join(opt_dir, f"opt_mode{j:02d}_v{v}.gif"), frames=30, interval=150, title=title)
        j += 1
