"""Linear elastic eigenvalue problem with mesh file input (free boundary)."""

from skfem import *
from skfem.helpers import dot, ddot, sym_grad, eye, trace
import numpy as np
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
import csv
import matplotlib.animation as animation

def von_mises_stress(sigma):
    """ミーゼス応力を計算する関数"""
    s_xx = sigma[0]
    s_yy = sigma[1]
    s_xy = sigma[2]
    return np.sqrt(s_xx**2 - s_xx * s_yy + s_yy**2 + 3 * s_xy**2)

class EigenvalueSolver:
    """2次元静弾性問題の固有値問題を解き、結果を可視化・出力するクラス (自由端)"""
    def __init__(self, mesh_filename, lam=1.0, mu=1.0, rho=1.0, sigma=0.0, n_eig=6):
        """
        Args:
            mesh_filename (str): メッシュファイル名 (.msh)
            lam (float): ラメ定数
            mu (float): せん断弾性率
            n_eig (int): 求める固有値の数
        """
        self.mesh_filename = mesh_filename
        self.lam = lam
        self.mu = mu
        self.rho = rho
        self.sigma = sigma
        self.n_eig = n_eig
        self.mesh = None
        self.basis = None
        self.L = None
        self.x = None

    def C(self, T):
        """構成テンソルを計算する"""
        return 2. * self.mu * T + self.lam * eye(trace(T), T.shape[0])

    def solve(self):
        """固有値問題を解く"""
        try:
            # メッシュの読み込み
            self.mesh = MeshTri.load(self.mesh_filename)
            e1 = ElementTriP2() # 2次要素; default
            e = ElementVector(e1)
            self.basis = Basis(self.mesh, e, intorder=2) # default

            @BilinearForm
            def stiffness(u, v, w):
                return ddot(self.C(sym_grad(u)), sym_grad(v))

            @BilinearForm
            def mass(u, v, w):
                return self.rho * dot(u, v)

            K = stiffness.assemble(self.basis)
            M = mass.assemble(self.basis)

            self.L, self.x = eigsh(K, k=self.n_eig, M=M, sigma=self.sigma, which='LM') # IRAM(?)
            return self.L, self.x

        except RuntimeError as e:
            if "Factor is exactly singular" in str(e):
                print("Error: Matrix is singular.")
                return False
            else:
                raise e  # その他のRuntimeErrorは再送出
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return False

    def visualize_mode(self, mode_index, scale=0.1, filename="mode.png", title=""):
        """指定した固有モードの変形形状とミーゼス応力の画像を生成する"""
        if self.L is None or self.x is None or self.mesh is None or self.basis is None:
            print("Error: 固有値問題を解いてから可視化を実行してください。")
            return None

        if not 0 <= mode_index < self.n_eig:
            print(f"Error: モードインデックス {mode_index} は有効な範囲外です (0 から {self.n_eig - 1} の間)。")
            return None

        from skfem.visuals.matplotlib import plot, draw

        y = self.x[:, mode_index]
        sbasis = self.basis.with_element(ElementVector(self.basis.elem))
        yi = self.basis.interpolate(y)
        sigma = sbasis.project(self.C(sym_grad(yi)))  # self.C を使用

        # ミーゼス応力を計算
        vm = von_mises_stress(sigma[sbasis.nodal_dofs])

        # メッシュの高さ（Y方向の範囲）を計算
        mesh_height = np.max(self.mesh.p[1]) - np.min(self.mesh.p[1])

        # Y座標を反転して高さを足す
        inverted_p = self.mesh.p.copy()
        inverted_p[1] *= -1
        inverted_p[1] += mesh_height  # 高さを足す

        inverted_y = y[self.basis.nodal_dofs].copy()
        inverted_y[1::2] *= -1  # Y変位成分の符号を反転

        M_disp = MeshTri(np.array(inverted_p + scale * inverted_y), self.mesh.t)

        fig, ax = plt.subplots(figsize=(6, 5))
        draw(M_disp, ax=ax)
        plot(M_disp, vm, ax=ax, colorbar='Von Mises Stress', shading='gouraud') # ミーゼス応力をプロット
        #ax.set_title(f"Mode {mode_index+1}, Eigenvalue: {self.L[mode_index]:.4e}")
        ax.set_title(title)
        ax.set_aspect('equal')  # 縦横比を同じにする
        ax.grid(True) # グリッド線を追加
            
        plt.tight_layout()
        plt.savefig(filename)
        print(f"モード {mode_index+1} の画像を {filename} に保存しました。")
        plt.close(fig)
        return fig        

    def visualize_all_modes(self, scale=0.1):
        """全ての固有モードの変形形状とミーゼス応力を可視化する"""
        if self.L is None or self.x is None or self.mesh is None or self.basis is None:
            print("Error: 固有値問題を解いてから可視化を実行してください。")
            return None

        from skfem.visuals.matplotlib import plot, draw

        # サブプロットのレイアウトを決定
        cols = int(np.ceil(np.sqrt(self.n_eig)))
        rows = int(np.ceil(self.n_eig / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))  # サブプロットを作成
        axes = np.array(axes).flatten()  # 2次元配列を1次元配列に変換

        # メッシュの高さ（Y方向の範囲）を計算
        mesh_height = np.max(self.mesh.p[1]) - np.min(self.mesh.p[1])

        for i in range(self.n_eig):
            y = self.x[:, i]
            sbasis = self.basis.with_element(ElementVector(self.basis.elem))
            yi = self.basis.interpolate(y)
            sigma = sbasis.project(self.C(sym_grad(yi)))  # self.C を使用

            # ミーゼス応力を計算
            vm = von_mises_stress(sigma[sbasis.nodal_dofs])

            # Y座標を反転して高さを足す
            inverted_p = self.mesh.p.copy()
            inverted_p[1] *= -1
            inverted_p[1] += mesh_height  # 高さを足す

            inverted_y = y[self.basis.nodal_dofs].copy()
            inverted_y[1::2] *= -1  # Y変位成分の符号を反転

            M_disp = MeshTri(np.array(inverted_p + scale * inverted_y), self.mesh.t)
            ax = axes[i]
            draw(M_disp, ax=ax)
            plot(M_disp, vm, ax=ax, colorbar='Von Mises Stress', shading='gouraud') # ミーゼス応力をプロット
            ax.set_title(f"Mode {i+1}, Eigenvalue: {self.L[i]:.1f}")
            ax.set_aspect('equal')  # 縦横比を同じにする

            
        # 余分なサブプロットを非表示にする
        for j in range(self.n_eig, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()  # サブプロット間のスペースを調整
        return fig


    def animate_mode(self, mode_index, v, filename="mode_animation.gif", frames=50, interval=200, title=""):
        """指定した固有モードのアニメーションGIFを生成する (scale: 0, v, 0, -v, 0)"""
        if self.L is None or self.x is None or self.mesh is None or self.basis is None:
            print("Error: 固有値問題を解いてからアニメーションを実行してください。")
            return None

        if not 0 <= mode_index < self.n_eig:
            print(f"Error: モードインデックス {mode_index} は有効な範囲外です (0 から {self.n_eig - 1} の間)。")
            return None

        from skfem.visuals.matplotlib import plot, draw
        import matplotlib.cm as cm

        y = self.x[:, mode_index]
        sbasis = self.basis.with_element(ElementVector(self.basis.elem))
        yi = self.basis.interpolate(y)
        sigma = sbasis.project(self.C(sym_grad(yi)))  # self.C を使用
        vm = von_mises_stress(sigma[sbasis.nodal_dofs])

        # Figure のサイズを固定
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.set_aspect('equal')
        #ax.set_title(f"Mode {mode_index+1}, Eigenvalue: {self.L[mode_index]:.4e}")
        ax.set_title(title)
        colorbar_obj = None
        plot_obj = None
        cmap = cm.viridis

        # メッシュの Y 方向の範囲
        mesh_min_y = np.min(self.mesh.p[1])
        mesh_max_y = np.max(self.mesh.p[1])
        mesh_height = mesh_max_y - mesh_min_y

        # Y 変位の最大振幅
        max_dy = np.max(np.abs(y[self.basis.nodal_dofs][1::2])) * v

        # 固定の描画範囲を計算 (Y軸をわずかに広げる)
        padding_y = 0.1 * mesh_height  # Y軸の範囲に少し余裕を持たせる
        min_x = np.min(self.mesh.p[0]) - np.max(np.abs(y[self.basis.nodal_dofs][0::2])) * v
        max_x = np.max(self.mesh.p[0]) + np.max(np.abs(y[self.basis.nodal_dofs][0::2])) * v
        min_y_transformed = -(mesh_max_y) - max_dy + mesh_height - padding_y
        max_y_transformed = -(mesh_min_y) + max_dy + mesh_height + padding_y

        def update(frame):
            nonlocal colorbar_obj
            nonlocal plot_obj
            scales = [0, v, 0, -v, 0]
            scale_index = frame % len(scales)
            current_scale = scales[scale_index]

            # Y座標を反転して高さを足す
            inverted_p = self.mesh.p.copy()
            inverted_p[1] *= -1
            inverted_p[1] += mesh_height  # 高さを足す

            inverted_y = y[self.basis.nodal_dofs].copy()
            inverted_y[1::2] *= -1  # Y変位成分の符号を反転

            M_disp = MeshTri(np.array(inverted_p + current_scale * inverted_y), self.mesh.t)
            ax.clear()
            ax.set_aspect('equal')
            #ax.set_title(f"Mode {mode_index+1}, Eigenvalue: {self.L[mode_index]:.4e}")
            ax.set_title(title)
            ax.set_xlim(min_x, max_x)
            ax.set_ylim(min_y_transformed, max_y_transformed) # Y軸の範囲を再設定
            draw(M_disp, ax=ax)
            plot_object = plot(M_disp, vm, ax=ax, colorbar=False, shading='gouraud', cmap=cmap)
            if plot_object and hasattr(plot_object, 'collections') and plot_object.collections:
                mappable = plot_object.collections[0]
                if colorbar_obj is None:
                    colorbar_obj = fig.colorbar(mappable, ax=ax, label='Von Mises Stress')
            return plot_object,

        plt.tight_layout()
        ani = animation.FuncAnimation(fig, update, frames=frames, interval=interval, blit=False)
        ani.save(filename, writer='pillow')
        print(f"モード {mode_index+1} のアニメーションを {filename} に保存しました。")
        plt.close(fig)
        return ani

    
    
    def write_mode_data(self, mode_index, filename_base="mode"):
        """指定された固有モードのミーゼス応力と変位をファイルに書き込む"""
        if self.L is None or self.x is None or self.mesh is None or self.basis is None:
            print("Error: 固有値問題を解いてからデータ書き込みを実行してください。")
            return

        if not 0 <= mode_index < self.n_eig:
            print(f"Error: モードインデックス {mode_index} は有効な範囲外です (0 から {self.n_eig - 1} の間)。")
            return

        y = self.x[:, mode_index]
        sbasis = self.basis.with_element(ElementVector(self.basis.elem))
        yi = self.basis.interpolate(y)
        sigma = sbasis.project(self.C(sym_grad(yi))) # self.C を使用
        vm = von_mises_stress(sigma[sbasis.nodal_dofs])
        displacement = y[self.basis.nodal_dofs].reshape((2, -1)).T # (n_nodes, 2) の形状に変形

        # 変位をCSVファイルに書き込む
        displacement_filename = f"{filename_base}_{mode_index + 1}_displacement.csv"
        with open(displacement_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Node ID', 'X Displacement', 'Y Displacement'])
            for i, disp in enumerate(displacement):
                writer.writerow([i, disp[0], disp[1]])
        print(f"変位データを {displacement_filename} に書き込みました。")

        # ミーゼス応力をCSVファイルに書き込む
        von_mises_filename = f"{filename_base}_{mode_index + 1}_von_mises.csv"
        with open(von_mises_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Node ID', 'Von Mises Stress'])
            for i, stress in enumerate(vm):
                writer.writerow([i, stress])
        print(f"ミーゼス応力データを {von_mises_filename} に書き込みました。")

if __name__ == '__main__':
     # 例として、'square.msh' というメッシュファイルが存在する場合
    mesh_file = 'square.msh'
    solver = EigenvalueSolver(mesh_file, n_eig=6)
    results = solver.solve()

    if results:
        eigenvalues, eigenvectors = results
        print("Eigenvalues:\n", eigenvalues)

        # 全てのモードを可視化
        fig = solver.visualize_all_modes(scale=0.5)
        if fig:
            plt.show()

        # 特定のモードのデータを書き出す (例: 1番目のモード)
        solver.write_mode_data(0)
        solver.write_mode_data(2, filename_base="mode_output")


# from skfem import *
# from skfem.helpers import dot, ddot, sym_grad, eye, trace
# import numpy as np
# #from scipy.sparse.linalg import eigsh
# from scipy.linalg import eigh  # eigh をインポート
# from scipy.sparse.linalg import eigsh  # eigh をインポート
# import matplotlib.pyplot as plt
# 
# 
# def von_mises_stress(sigma):
#     """ミーゼス応力を計算する関数"""
#     s_xx = sigma[0]
#     s_yy = sigma[1]
#     s_xy = sigma[2]
#     return np.sqrt(s_xx**2 - s_xx * s_yy + s_yy**2 + 3 * s_xy**2)
# 
# def solve_eigenvalue_problem(mesh_filename, lam=1.0, mu=1.0, rho=1.0, sigma=0.0, n_eig=6):
#     """
#     scikit-fem用のメッシュファイルを入力として、2次元静弾性問題の固有値問題を解くプログラム (自由端)
# 
#     Args:
#         mesh_filename (str): メッシュファイル名 (.msh)
#         lam (float): ラメ定数
#         mu (float): せん断弾性率
#         n_eig (int): 求める固有値の数
#     """
# 
#     try:
#         # メッシュの読み込み
#         mesh = MeshTri.load(mesh_filename)
#         e1 = ElementTriP2() # 2次要素; default
#         #e1 = ElementTriP1() # 1次要素
# 
#         # 要素の定義
#         e = ElementVector(e1)
# 
#         #basis = Basis(mesh, e, intorder=3)
#         basis = Basis(mesh, e, intorder=2) # default
#         #basis = Basis(mesh, e, intorder=1)
# 
#         def C(T):
#             return 2. * mu * T + lam * eye(trace(T), T.shape[0])
# 
#         @BilinearForm
#         def stiffness(u, v, w):
#             return ddot(C(sym_grad(u)), sym_grad(v))
# 
#         @BilinearForm
#         def mass(u, v, w):
#             return rho * dot(u, v)
# 
#         K = stiffness.assemble(basis)
#         M = mass.assemble(basis)
# 
#         L, x = eigsh(K, k=n_eig, M=M, sigma=sigma, which='LM') # IRAM(?)
#         #L, x = eigh(K.toarray(), b=M.toarray()) # LAPACK
# 
#         def visualize_all_modes(scale=0.1):
#             from skfem.visuals.matplotlib import plot, draw
# 
#             # サブプロットのレイアウトを決定
#             cols = int(np.ceil(np.sqrt(n_eig)))
#             rows = int(np.ceil(n_eig / cols))
# 
#             fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))  # サブプロットを作成
#             axes = np.array(axes).flatten()  # 2次元配列を1次元配列に変換
# 
#             for i in range(n_eig):
#                 y = x[:, i]
#                 sbasis = basis.with_element(ElementVector(e))
#                 yi = basis.interpolate(y)
#                 sigma = sbasis.project(C(sym_grad(yi)))
# 
#                 # ミーゼス応力を計算
#                 von_mises = von_mises_stress(sigma[sbasis.nodal_dofs])
# 
#                 M_disp = MeshTri(np.array(mesh.p + scale * y[basis.nodal_dofs]), mesh.t)
#                 ax = axes[i]
#                 draw(M_disp, ax=ax)
#                 plot(M_disp, von_mises, ax=ax, colorbar='Von Mises Stress', shading='gouraud') # ミーゼス応力をプロット
#                 ax.set_title(f"Mode {i+1}, Eigenvalue: {L[i]:.4e}")
#                 ax.set_aspect('equal')  # 縦横比を同じにする
# 
#             # 余分なサブプロットを非表示にする
#             for j in range(n_eig, len(axes)):
#                 axes[j].axis('off')
# 
#             plt.tight_layout()  # サブプロット間のスペースを調整
#             return fig
# 
#         return L, x, visualize_all_modes
# 
#     except RuntimeError as e:
#         if "Factor is exactly singular" in str(e):
#             print("Error: Matrix is singular.")
#             return False
#         else:
#             raise e  # その他のRuntimeErrorは再送出
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")
#         return False
# 
# 
#     
# if __name__ == "__main__":
#     # メッシュファイル名
#     #mesh_filename = "100.msh"
#     #mesh_filename = "circle.msh" # Shiと比較するならば円形；メッシュ数はcircle.geoで変更する
#     #mesh_filename = "ring.msh"
#     #mesh_filename = "circle_sym.msh" # だめである。singularになる。
#     msh_path = "test/gmsh/output_00001.msh"
#     
#     # 固有値問題の求解 (n_eig を任意の値に変更可能)
#     n_eig = 16
#     #eigenvalues, eigenmodes, visualize_all_modes = solve_eigenvalue_problem(mesh_filename, n_eig=n_eig)
# 
#     E = 2.0*10**11
#     nu = 0.3
#     lam = E * nu / ((1.0+nu)*(1.0-2.0*nu)) # https://ja.wikipedia.org/wiki/%E5%BC%BE%E6%80%A7%E7%8E%87
#     mu = E / (2.0*(1.0+nu))
#     rho = 7800.0
#     
#     Es = E / (1.0-nu**2)
#     nus = nu / (1.0-nu)
#     unit = np.sqrt(E/(rho*(1-nu**2))) # こちらを単位とするとき、平面応力状態であるShiの結果に近い；つまり、本計算は平面応力状態
#     print("unit=", unit)
#     units = np.sqrt(Es/(rho*(1-nus**2)))
#     print("units=", units)
#     
#     sigma=0.0 # default
#     #sigma=1.8*unit
#     #sigma=2.0*unit
# 
#     #eigenvalues, eigenmodes, visualize_all_modes = solve_eigenvalue_problem(mesh_filename, lam=lam, mu=mu, rho=rho, sigma=sigma, n_eig=n_eig)
#     result = solve_eigenvalue_problem(msh_path, lam=lam, mu=mu, rho=rho, sigma=sigma, n_eig=n_eig)
#     if result is False:
#         print("固有値問題の解決に失敗しました。メッシュファイルを確認してください。")
#     else:
#         eigenvalues, eigenmodes, visualize_all_modes = result
#         # 固有値と固有モードを使用する処理
#         print("固有値:", eigenvalues)
#         # visualize_all_modes() # 可視化関数を実行する場合
#     
#         # 固有値の表示
#         print("Eigenvalues:")
#         i=0
#         for eigenvalue in eigenvalues:
#             omega = np.sqrt(eigenvalue)
#             print(f"{i+1} {eigenvalue:.4e} {omega:.4e} {omega/unit:.4e} {omega/units:.4e}")
#             i=i+1
#         # 全てのモードを可視化
#         visualize_all_modes().show()
#         plt.pause(0)  # ウィンドウを表示したままにする
# 
