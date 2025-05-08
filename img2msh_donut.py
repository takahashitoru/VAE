import cv2
import numpy as np
import gmsh
import os
import shutil

from resolve_self_intersection import resolve_self_intersection
import fixgeo
#from fix_self_intersection import fix_self_intersected_line
from fix_self_intersection import fix_self_intersecting_lines
from remove_overlapping_lines import remove_overlapping_lines


def img2msh_donut(image_file, msh_file, max_area=None, min_angle=None, magnification=1.0, smooth_kernel_size=0, min_contour_area=100):
    """
    輪郭を境界としてgmshのgeoフォーマットで保存し、msh形式に変換します。
    ドーナツ型と複数の独立した領域に対応します。

    Args:
        image_file (str): 入力画像のパス。
        msh_file (str): 結果を保存するmshファイルのパス。
        max_area (float, optional): 最大要素面積。
        min_angle (float, optional): 最小要素角度。
        magnification (float, optional): 画像の拡大率。
        smooth_kernel_size (int, optional): スムージングに使用するガウシアンフィルタのカーネルサイズ。
        min_contour_area (int, optional): 無視する最小の輪郭面積。
    """
    # 画像を読み込み、グレースケールに変換
    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # アップサンプリング
    if magnification > 1:
        height, width = gray.shape
        new_height, new_width = int(height * magnification), int(width * magnification)
        gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # スムージング処理 (ガウシアンフィルタ)
    if smooth_kernel_size > 0:
        gray = cv2.GaussianBlur(gray, (smooth_kernel_size, smooth_kernel_size), 0)

    # 外周を白色で埋める
    gray[0, :] = 255
    gray[-1, :] = 255
    gray[:, 0] = 255
    gray[:, -1] = 255

    # 二値化
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # 輪郭抽出 (RETR_CCOMPを使用)
    #contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    print(f"処理ファイル: {image_file}")
    print(f"検出された輪郭の数: {len(contours)}")
    print(f"階層構造: {hierarchy}")

    processed_contours = []
    original_indices = []
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area >= min_contour_area:
            processed_contours.append(np.array(contour))
            original_indices.append(i)

    contours = processed_contours
    line_index = 1
    vertex_index = 1
    loop_index = 1

    geo_tmpfile = msh_file.replace('.msh', '_tmp.geo')
    with open(geo_tmpfile, 'w') as f:
        point_map = {}
        contour_loop_map = {}
        surface_definitions = {}
        loop_tags = {}

        # 輪郭からLine Loopを生成
        for i, contour in enumerate(contours):
            line_tags = []
            for j in range(len(contour)):
                p1_tuple = tuple(contour[j][0])
                p2_tuple = tuple(contour[(j + 1) % len(contour)][0])

                if p1_tuple not in point_map:
                    point_map[p1_tuple] = vertex_index
                    f.write(f'Point({vertex_index}) = {{{p1_tuple[0]}, {p1_tuple[1]}, 0}};\n')
                    v1 = vertex_index
                    vertex_index += 1
                else:
                    v1 = point_map[p1_tuple]

                if p2_tuple not in point_map:
                    point_map[p2_tuple] = vertex_index
                    f.write(f'Point({vertex_index}) = {{{p2_tuple[0]}, {p2_tuple[1]}, 0}};\n')
                    v2 = vertex_index
                    vertex_index += 1
                else:
                    v2 = point_map[p2_tuple]

                f.write(f'Line({line_index}) = {{{v1}, {v2}}};\n')
                line_tags.append(line_index)
                line_index += 1

            loop_tag = loop_index
            f.write(f'Line Loop({loop_tag}) = {{{", ".join(map(str, line_tags))}}};\n')
            contour_loop_map[original_indices[i]] = loop_tag
            loop_tags[original_indices[i]] = loop_tag
            loop_index += 1

        surface_index = 1
        processed_parents = set()

        if hierarchy is not None and len(hierarchy) > 0:
            for i in range(len(contours)):
                parent_index = hierarchy[0][original_indices[i]][3]
                current_index = original_indices[i]
                current_loop = contour_loop_map[current_index]

                if parent_index == -1:
                    # 親を持たない輪郭は外側の輪郭
                    surface_definitions[surface_index] = [current_loop]
                    surface_index += 1
                elif parent_index in original_indices:
                    # 親を持つ輪郭は内側の穴
                    parent_surface_key = None
                    parent_original_index = parent_index
                    for s_idx, loops in surface_definitions.items():
                        parent_loop = contour_loop_map[parent_original_index]
                        if parent_loop in loops and parent_index not in processed_parents:
                            parent_surface_key = s_idx
                            break
                        elif parent_loop in loops:
                            parent_surface_key = s_idx
                            break

                    if parent_surface_key is not None:
                        surface_definitions[parent_surface_key].append(-current_loop)

            # Plane Surface を書き出す
            for s_idx, loops in surface_definitions.items():
                f.write(f'Plane Surface({s_idx}) = {{{", ".join(map(str, loops))}}};\n')

        elif contour_loop_map:
            for loop_idx in contour_loop_map.values():
                f.write(f'Plane Surface({surface_index}) = {{{loop_idx}}};\n')
                surface_index += 1

    # geoファイルの処理
    """
    print("###### fixgeo.process_geo_file ######")
    geo_tmpfile2 = msh_file.replace('.msh', '_tmp2.geo')
    shutil.copyfile(geo_tmpfile, geo_tmpfile2) # just a copy

    print("###### remove_overlapping_lines ######")
    geo_tmpfile3 = msh_file.replace('.msh', '_tmp3.geo')
    remove_overlapping_lines(geo_tmpfile2, geo_tmpfile3)

    print("###### fix_self_intersected_line ######")
    geo_tmpfile4 = msh_file.replace('.msh', '_tmp4.geo')
    #fixgeo.fix_self_intersected_line(geo_tmpfile3, geo_tmpfile4, scale=0.1)
    fix_self_intersected_line(geo_tmpfile3, geo_tmpfile4, scale=0.3) # iteration version
    """


    print("###### fix_self_intersecting_lines ######")
    geo_tmpfile2 = msh_file.replace('.msh', '_tmp2.geo')
    #fixgeo.fix_self_intersected_line(geo_tmpfile, geo_tmpfile2, scale=0.1)
    fix_self_intersecting_lines(geo_tmpfile, geo_tmpfile2, scale=0.3) # iteration version

    #print("###### remove_overlapping_lines ######")
    #geo_tmpfile4 = msh_file.replace('.msh', '_tmp4.geo')
    #remove_overlapping_lines(geo_tmpfile3, geo_tmpfile4)

    print("###### resolve_self_intersection ######")
    geo_file = msh_file.replace('.msh', '.geo')
    resolve_self_intersection(geo_tmpfile2, geo_file)

    # gmshの処理 (変更なし)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)

    try:
        gmsh.open(geo_file)
    except Exception as e:
        print(f"gmsh error: {e}")
        gmsh.finalize()
        return False

    if max_area is not None:
        gmsh.option.setNumber("Mesh.MeshSizeMax", max_area)
    if min_angle is not None:
        gmsh.option.setNumber("Mesh.MinimumCircleRadiusFactor", np.sin(np.radians(min_angle)) / np.sin(np.radians(180 - 2 * min_angle)))

    try:
        gmsh.model.mesh.generate(2)
    except Exception as e:
        print(f"gmsh error: {e}")
        gmsh.finalize()
        return False

    gmsh.write(msh_file)
    gmsh.finalize()

    return True


if __name__ == '__main__':

    # ドーナツ型の画像を生成
    height, width = 200, 200
    image_donut = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.circle(image_donut, (width // 2, height // 2), 80, (0, 0, 0), -1)
    cv2.circle(image_donut, (width // 2, height // 2), 40, (255, 255, 255), -1)
    pngfile_donut = os.path.join("./", "donut_test.png")
    cv2.imwrite(pngfile_donut, image_donut)
    mshfile_donut = pngfile_donut.replace('.png', '.msh')
    img2msh_donut(pngfile_donut, mshfile_donut, min_contour_area=100)

    # 単純な四角形の画像を生成 (テスト用)
    height_square, width_square = 100, 100
    image_square = np.zeros((height_square, width_square, 3), dtype=np.uint8)
    cv2.rectangle(image_square, (20, 20), (80, 80), (255, 255, 255), -1)
    pngfile_square = os.path.join("./", "square_test.png")
    cv2.imwrite(pngfile_square, image_square)
    mshfile_square = pngfile_square.replace('.png', '.msh')
    img2msh_donut(pngfile_square, mshfile_square, min_contour_area=100)    

    # 円&donutの画像を生成
    height, width = 200, 200
    image_multi = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.circle(image_multi, (width // 4, height // 4), 40, (0, 0, 0), -1)
    cv2.circle(image_multi, ((width * 3) // 4, (height * 3) // 4), 40, (0, 0, 0), -1)
    cv2.circle(image_multi, ((width * 3) // 4, (height * 3) // 4), 20, (255, 255, 255), -1)
    pngfile_multi = os.path.join("./", "cd_test.png")
    cv2.imwrite(pngfile_multi, image_multi)
    mshfile_multi = pngfile_multi.replace('.png', '.msh')
    img2msh_donut(pngfile_multi, mshfile_multi, min_contour_area=100)
    
