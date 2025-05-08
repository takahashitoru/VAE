import math

def rotate_vector_clockwise(x, y):
     """ベクトルを時計回りに90度回転させる"""
     return y, -x
 
def normalize_vector(x, y):
    """ベクトルを正規化する"""
    magnitude = math.sqrt(x**2 + y**2)
    if magnitude == 0:
        return 0, 0
    return x / magnitude, y / magnitude


#def process_geo(input_geo_path, output_geo_path, scale=0.1):
def fix_self_intersected_line(input_geo_path, output_geo_path, scale=0.1):
    """自己交差するLineを修正し、新しいGEOファイルを出力する (デバッグ情報を含む)"""
    try:
        with open(input_geo_path, 'r') as f:
            input_geo_content = f.read()
    except FileNotFoundError:
        print(f"エラー: 入力ファイル '{input_geo_path}' が見つかりません。")
        return

    lines = []
    points = {}
    line_loops = {}
    plane_surfaces = {}
    max_point_index = 0
    max_line_index = 0

    # 入力GEOファイルを解析
    for line_str in input_geo_content.strip().split('\n'):
        line_str = line_str.strip()
        if line_str.startswith("Point("):
            #print(f"解析中の行 (Point): '{line_str}'")
            parts = line_str[len("Point("):].split(")")
            #print(f"parts (Point): {parts}")
            if len(parts) > 1:
                try:
                    index_str = parts[0].strip()
                    index = int(index_str)
                    coords_part = parts[1].strip() # 先頭と末尾の空白を削除
                    if coords_part.startswith("=") and coords_part[coords_part.find("=") + 1:].lstrip().startswith("{"): # '=' で始まり、その後に '{' が続くかチェック
                        coords_str = coords_part[coords_part.find("=") + 1:].lstrip().strip('{}').rstrip(';') # '=' 以降の空白と '{}' を削除し、末尾の ';' を削除
                        #print(f"coords_str (Point, 処理後): '{coords_str}'")
                        coords_list_str = coords_str.split(',')
                        #print(f"coords_list_str (Point, split後): {coords_list_str}")
                        coords = []
                        for c_str in coords_list_str:
                            cleaned_c = c_str.strip().rstrip('}') # 末尾の '}' を削除
                            float_c = float(cleaned_c)
                            coords.append(float_c)
                        #print(f"coords (Point, float変換後): {coords}")
                        points[index] = coords
                        max_point_index = max(max_point_index, index)
                    else:
                        print("警告 (Point): 座標部分の形式が不正です")
                except ValueError as e:
                    print(f"ValueError (Point) が発生しました: {e}")
                    print(f"変換に失敗した文字列 (Point): '{cleaned_c}'")
                    raise
                except Exception as e:
                    print(f"予期しないエラー (Point) が発生しました: {e}")
                    raise
            else:
                print("警告 (Point): 行の形式が不正です")
        elif line_str.startswith("Line("):
            #print(f"解析中の行 (Line): '{line_str}'")
            parts = line_str[len("Line("):].split(")")
            #print(f"parts (Line): {parts}")
            if len(parts) > 1:
                try:
                    index_str = parts[0].strip()
                    index = int(index_str)
                    points_part = parts[1].strip() # 先頭と末尾の空白を削除
                    if points_part.startswith("=") and points_part[points_part.find("=") + 1:].lstrip().startswith("{"): # '=' で始まり、その後に '{' が続くかチェック
                        point_indices_str = points_part[points_part.find("=") + 1:].lstrip().strip('{}').split(',') # '=' 以降の空白と '{}' を削除
                        point_indices = tuple(int(p.strip().rstrip(';}')) for p in point_indices_str) # 各インデックスから末尾の ';' と '}' を削除
                        lines.append({'index': index, 'points': point_indices})
                        max_line_index = max(max_line_index, index)
                    else:
                        print("警告 (Line): 接続点インデックス部分の形式が不正です")
                except ValueError as e:
                    print(f"ValueError (Line) が発生しました: {e}")
                    raise
                except Exception as e:
                    print(f"予期しないエラー (Line) が発生しました: {e}")
                    raise
            else:
                print("警告 (Line): 行の形式が不正です")
        elif line_str.startswith("Line Loop("):
            #print(f"解析中の行 (Line Loop): '{line_str}'")
            parts = line_str[len("Line Loop("):].split(")")
            #print(f"parts (Line Loop): {parts}")
            if len(parts) > 1:
                try:
                    index_str = parts[0].strip()
                    index = int(index_str)
                    lines_part = parts[1].strip() # 先頭と末尾の空白を削除
                    if lines_part.startswith("=") and lines_part[lines_part.find("=") + 1:].lstrip().startswith("{"): # '=' で始まり、その後に '{' が続くかチェック
                        line_indices_str = lines_part[lines_part.find("=") + 1:].lstrip().strip('{}').split(',') # '=' 以降の空白と '{}' を削除
                        line_indices = [int(l.strip().rstrip(';}')) for l in line_indices_str] # 各インデックスから末尾の ';' と '}' を削除
                        line_loops[index] = line_indices
                    else:
                        print("警告 (Line Loop): 線インデックス部分の形式が不正です")
                except ValueError as e:
                    print(f"ValueError (Line Loop) が発生しました: {e}")
                    raise
                except Exception as e:
                    print(f"予期しないエラー (Line Loop) が発生しました: {e}")
                    raise
            else:
                print("警告 (Line Loop): 行の形式が不正です")
        elif line_str.startswith("Plane Surface("):
            #print(f"解析中の行 (Plane Surface): '{line_str}'")
            parts = line_str[len("Plane Surface("):].split(")")
            #print(f"parts (Plane Surface): {parts}")
            if len(parts) > 1:
                try:
                    index_str = parts[0].strip()
                    index = int(index_str)
                    loops_part = parts[1].strip() # 先頭と末尾の空白を削除
                    if loops_part.startswith("=") and loops_part[loops_part.find("=") + 1:].lstrip().startswith("{"): # '=' で始まり、その後に '{' が続くかチェック
                        loop_indices_str = loops_part[loops_part.find("=") + 1:].lstrip().strip('{}').split(',') # '=' 以降の空白と '{}' を削除
                        loop_indices = [int(l.strip().rstrip(';}')) for l in loop_indices_str] # 各インデックスから末尾の ';' と '}' を削除
                        plane_surfaces[index] = loop_indices
                    else:
                        print("警告 (Plane Surface): ループインデックス部分の形式が不正です")
                except ValueError as e:
                    print(f"ValueError (Plane Surface) が発生しました: {e}")
                    raise
                except Exception as e:
                    print(f"予期しないエラー (Plane Surface) が発生しました: {e}")
                    raise
            else:
                print("警告 (Plane Surface): 行の形式が不正です")

    new_points = points.copy()
    updated_lines = list(lines) # Lineのリストをコピーして変更を加える
    modified_m_index = None
    original_m_points = None
    point_replacement_m = {}
    next_point_index = max_point_index + 1
    line_l_index = None
    line_l_points = None # Line L の端点を保持

    # 自己交差するLineのペアを探索と修正 (Line M の情報を保持)
    for i in range(len(lines)):
        line_l = lines[i]
        for j in range(i + 1, len(lines)):
            line_m = lines[j]
            if set(line_l['points']) == set(line_m['points']) and line_l['points'] != line_m['points']:
                line_l_index = line_l['index']
                point_a_l, point_b_l = line_l['points']
                line_l_points = (point_a_l, point_b_l) # Line L の端点を記憶
                a_coords = new_points[point_a_l]
                b_coords = new_points[point_b_l]

                # ベクトル AB (Line L)
                ab_x_l = b_coords[0] - a_coords[0]
                ab_y_l = b_coords[1] - a_coords[1]
                line_length_l = math.sqrt(ab_x_l**2 + ab_y_l**2)

                # 時計回りに90度回転させて正規化 (法線ベクトル)
                normal_x, normal_y = normalize_vector(*rotate_vector_clockwise(ab_x_l, ab_y_l))

                # 新しい点の座標を計算 (Line L からのオフセット)
                offset = line_length_l * scale
                c_coords = [a_coords[0] + normal_x * offset, a_coords[1] + normal_y * offset, 0.0]
                d_coords = [b_coords[0] + normal_x * offset, b_coords[1] + normal_y * offset, 0.0]

                new_points[next_point_index] = c_coords
                point_c = next_point_index
                next_point_index += 1
                new_points[next_point_index] = d_coords
                point_d = next_point_index

                # Line M の情報を記録
                modified_m_index = line_m['index']
                original_m_points = line_m['points']
                point_replacement_m[original_m_points[0]] = point_d
                point_replacement_m[original_m_points[1]] = point_c

                # updated_lines 内の Line M を更新
                for k, updated_line in enumerate(updated_lines):
                    if updated_line['index'] == line_m['index']:
                        print(f"デバッグ: Line {updated_line['index']} を ({point_d}, {point_c}) に更新")
                        updated_lines[k]['points'] = (point_d, point_c)
                        break
                break # 一つの自己交差ペアを処理したらループを抜ける

    # Lineに接続する点の情報を更新 (Line M に隣接する Line のみ)
    if modified_m_index is not None and original_m_points is not None and line_l_index is not None and line_l_points is not None:
        print(f"デバッグ: Line L (index {line_l_index}) の端点: {line_l_points}")
        print(f"デバッグ: Line M (index {modified_m_index}) の元の端点: {original_m_points}")
        print(f"デバッグ: 置換マップ: {point_replacement_m}")
        for i, updated_line in enumerate(updated_lines):
            current_line_index = updated_line['index']
            p1 = updated_line['points'][0]
            p2 = updated_line['points'][1]
            new_p1 = p1
            new_p2 = p2
            print(f"デバッグ: 検査中の Line {current_line_index} の端点: ({p1}, {p2})")

            # Line L とその隣接 Line はスキップ
            if current_line_index != line_l_index and current_line_index != line_l_index - 1 and current_line_index != line_l_index + 1:
                if p1 == original_m_points[0]:
                    new_p1 = point_replacement_m.get(p1, p1)
                    print(f"デバッグ: Line {current_line_index} の点 {p1} を {new_p1} に置換 (元のMの点0)")
                elif p1 == original_m_points[1]:
                    new_p1 = point_replacement_m.get(p1, p1)
                    print(f"デバッグ: Line {current_line_index} の点 {p1} を {new_p1} に置換 (元のMの点1)")

                if p2 == original_m_points[0]:
                    new_p2 = point_replacement_m.get(p2, p2)
                    print(f"デバッグ: Line {current_line_index} の点 {p2} を {new_p2} に置換 (元のMの点0)")
                elif p2 == original_m_points[1]:
                    new_p2 = point_replacement_m.get(p2, p2)
                    print(f"デバッグ: Line {current_line_index} の点 {p2} を {new_p2} に置換 (元のMの点1)")
            else:
                print(f"デバッグ: Line {current_line_index} は Line L またはその隣接 Line なのでスキップ")

            updated_lines[i]['points'] = (new_p1, new_p2)

    # Line Loop内のLineの参照を更新
    new_line_loops = {}
    for loop_index, loop_lines in line_loops.items():
        new_line_loops[loop_index] = list(loop_lines) # コピー

    # Plane Surface内のLine Loopの参照を更新
    new_plane_surfaces = {}
    for surface_index, surface_loops in plane_surfaces.items():
        new_plane_surfaces[surface_index] = list(surface_loops)

    # 新しいGEOファイルの内容を生成
    output_lines = []
    for index, coords in sorted(new_points.items()):
        output_lines.append(f"Point({index}) = {{{coords[0]}, {coords[1]}, {coords[2]}}};")

    for line in sorted(updated_lines, key=lambda x: x['index']):
        output_lines.append(f"Line({line['index']}) = {{{line['points'][0]}, {line['points'][1]}}};")

    for index, loop_lines in sorted(new_line_loops.items()):
        output_lines.append(f"Line Loop({index}) = {{{', '.join(map(str, loop_lines))}}};")

    for index, surface_loops in sorted(new_plane_surfaces.items()):
        output_lines.append(f"Plane Surface({index}) = {{{', '.join(map(str, surface_loops))}}};")

    output_geo_content = "\n".join(output_lines)

    # 新しいGEOファイルを出力
    try:
        os.makedirs(os.path.dirname(output_geo_path), exist_ok=True)
        with open(output_geo_path, 'w') as f:
            f.write(output_geo_content)
        print(f"修正後のGEOファイルは '{output_geo_path}' に保存されました。")
    except Exception as e:
        print(f"エラー: 出力ファイルへの書き込みに失敗しました: {e}")



# import re
# import os
# 
# def process_geo_file(input_file, output_file):
#     """
#     .geo ファイルを読み込み、同一の点から構成される線を削除し、
#     2つまたは1つの Line から構成される Line Loop を削除し、
#     閉じた線ループを正しく認識して .geo ファイルを出力する。
# 
#     Args:
#         input_file (str): 入力 .geo ファイルのパス
#         output_file (str): 出力 .geo ファイルのパス
#     """
# 
#     points = {}
#     lines = {}
# 
#     # .geo ファイルを読み込み、点と線の情報を抽出
#     with open(input_file, "r") as f:
#         for line in f:
#             line = line.strip()
#             if line.startswith("Point("):
#                 point_id, coords = parse_point(line)
#                 points[point_id] = coords
#             elif line.startswith("Line("):
#                 line_id, point_ids = parse_line(line)
#                 lines[line_id] = point_ids
# 
#     # 同一の点から構成される線を削除
#     lines = remove_duplicate_lines(lines)
# 
#     # 閉じた線ループを検出
#     line_loops = find_line_loops(lines)
# 
#     # 2つまたは1つの Line から構成される Line Loop を削除
#     line_loops, lines, points = remove_short_line_loops(line_loops, lines, points)
# 
#     # 出力 .geo ファイルを生成
#     geo_content = generate_geo_content(points, lines, line_loops)
# 
#     # 出力 .geo ファイルに書き込み
#     with open(output_file, "w") as f:
#         f.write(geo_content)
# 
#     print(f"{output_file} ファイルが生成されました。")
# 
# 
# def parse_point(line):
#     """
#     点の定義を解析する。
# 
#     Args:
#         line (str): 点の定義行 (例: "Point(1) = {35, 40, 0};")
# 
#     Returns:
#         tuple: 点のIDと座標
#     """
# 
#     parts = line.split("=")
#     point_id = int(parts[0].split("(")[1].split(")")[0])
#     coords = [float(c.strip()) for c in parts[1].strip("{} ;").split(",")]
#     return point_id, coords
# 
# 
# def parse_line(line):
#     """
#     線の定義を解析する。
# 
#     Args:
#         line (str): 線の定義行 (例: "Line(1) = {1, 2};")
# 
#     Returns:
#         tuple: 線のIDと接続する点のID
#     """
# 
#     parts = line.split("=")
#     line_id = int(parts[0].split("(")[1].split(")")[0])
#     point_ids = [int(p.strip()) for p in parts[1].strip("{} ;").split(",")]
#     return line_id, point_ids
# 
# 
# def remove_duplicate_lines(lines):
#     """
#     同一の点から構成される線を削除する。
# 
#     Args:
#         lines (dict): 線の接続情報
# 
#     Returns:
#         dict: 同一の点から構成される線を削除した後の線の接続情報
#     """
# 
#     return {line_id: point_ids for line_id, point_ids in lines.items() if point_ids[0] != point_ids[1]}
# 
# 
# def find_line_loops(lines):
#     """
#     線の情報から閉じた線ループを検出する。
# 
#     Args:
#         lines (dict): 線の接続情報 (例: {1: [1, 2], 2: [2, 3], ...})
# 
#     Returns:
#         list: 閉じた線ループのリスト (例: [[1, 2, 3, ...], [18, 19, 20, ...], ...])
#     """
# 
#     line_loops = []
#     visited_lines = set()
# 
#     for start_line_id in lines:
#         if start_line_id in visited_lines:
#             continue
# 
#         loop = [start_line_id]
#         visited_lines.add(start_line_id)
#         current_line_id = start_line_id
# 
#         while True:
#             next_line_id = find_next_line(lines, current_line_id, visited_lines)
#             if next_line_id is None:
#                 break
# 
#             loop.append(next_line_id)
#             visited_lines.add(next_line_id)
#             current_line_id = next_line_id
# 
#             if lines[current_line_id][1] == lines[start_line_id][0]:
#                 line_loops.append(loop)
#                 break
# 
#     return line_loops
# 
# 
# def find_next_line(lines, current_line_id, visited_lines):
#     """
#     次の線を検出する。
# 
#     Args:
#         lines (dict): 線の接続情報
#         current_line_id (int): 現在の線のID
#         visited_lines (set): 訪問済みの線のID
# 
#     Returns:
#         int: 次の線のID (見つからない場合はNone)
#     """
# 
#     current_end_point = lines[current_line_id][1]
# 
#     for line_id, point_ids in lines.items():
#         if line_id not in visited_lines and point_ids[0] == current_end_point:
#             return line_id
# 
#     return None
# 
# 
# def remove_short_line_loops(line_loops, lines, points):
#     """
#     2つまたは1つの Line から構成される Line Loop を削除する。
# 
#     Args:
#         line_loops (list): 線ループのリスト
#         lines (dict): 線の接続情報
#         points (dict): 点の座標情報
# 
#     Returns:
#         tuple: 削除後の線ループのリスト、線の接続情報、点の座標情報
#     """
# 
#     valid_line_loops = []
#     lines_to_remove = set()
#     points_to_remove = set()
# 
#     for loop in line_loops:
#         if len(loop) > 2:
#             valid_line_loops.append(loop)
#         else:
#             for line_id in loop:
#                 lines_to_remove.add(line_id)
#                 points_to_remove.add(lines[line_id][0])
#                 points_to_remove.add(lines[line_id][1])
# 
#     lines = {line_id: point_ids for line_id, point_ids in lines.items() if line_id not in lines_to_remove}
#     points = {point_id: coords for point_id, coords in points.items() if point_id not in points_to_remove}
# 
#     return valid_line_loops, lines, points
# 
# 
# def generate_geo_content(points, lines, line_loops):
#     """
#     .geo ファイルの内容を生成する。
# 
#     Args:
#         points (dict): 点の座標情報
#         lines (dict): 線の接続情報
#         line_loops (list): 閉じた線ループのリスト
# 
#     Returns:
#         str: .geo ファイルの内容
#     """
# 
#     geo_content = ""
# 
#     # 点の定義
#     for point_id, coord in points.items():
#         geo_content += f"Point({point_id}) = {{{coord[0]}, {coord[1]}, {coord[2]}}};\n"
# 
#     # 線の定義
#     for line_id, point_ids in lines.items():
#         geo_content += f"Line({line_id}) = {{{point_ids[0]}, {point_ids[1]}}};\n"
# 
#     # 線ループの定義
#     for i, loop in enumerate(line_loops):
#         geo_content += f"Line Loop({i + 1}) = {{{', '.join(map(str, loop))}}};\n"
#         geo_content += f"Plane Surface({i + 1}) = {{{i + 1}}};\n"
# 
#     return geo_content
        

import re
import os

def process_geo_file(input_file, output_file, largest_loop_only=False):
    """
    .geo ファイルを読み込み、同一の点から構成される線を削除し、
    2つまたは1つの Line から構成される Line Loop を削除し、
    閉じた線ループを正しく認識して .geo ファイルを出力する。
    largest_loop_only フラグが True の場合、線要素の数が最も多い線ループのみを生成する。

    Args:
        input_file (str): 入力 .geo ファイルのパス
        output_file (str): 出力 .geo ファイルのパス
        largest_loop_only (bool, optional): 最大の線ループのみを生成するかどうか (デフォルト: False)
    """

    points = {}
    lines = {}

    # .geo ファイルを読み込み、点と線の情報を抽出
    with open(input_file, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("Point("):
                point_id, coords = parse_point(line)
                points[point_id] = coords
            elif line.startswith("Line("):
                line_id, point_ids = parse_line(line)
                lines[line_id] = point_ids

    # 同一の点から構成される線を削除
    lines = remove_duplicate_lines(lines)

    # 閉じた線ループを検出
    line_loops = find_line_loops(lines)

    # 2つまたは1つの Line から構成される Line Loop を削除
    line_loops, lines, points = remove_short_line_loops(line_loops, lines, points)

    # largest_loop_only フラグが True の場合、最大の線ループのみを選択
    if largest_loop_only and line_loops:
        largest_loop = max(line_loops, key=len)
        line_loops = [largest_loop]
        lines_in_largest_loop = set(largest_loop)
        lines = {line_id: data for line_id, data in lines.items() if line_id in lines_in_largest_loop}
        points_in_largest_loop = set()
        for line_id in largest_loop:
            points_in_largest_loop.add(lines[line_id][0])
            points_in_largest_loop.add(lines[line_id][1])
        points = {point_id: data for point_id, data in points.items() if point_id in points_in_largest_loop}

    # 出力 .geo ファイルを生成
    geo_content = generate_geo_content(points, lines, line_loops)

    # 出力 .geo ファイルに書き込み
    with open(output_file, "w") as f:
        f.write(geo_content)

    print(f"{output_file} ファイルが生成されました。")


def parse_point(line):
    parts = line.split("=")
    point_id = int(parts[0].split("(")[1].split(")")[0])
    coords = [float(c.strip()) for c in parts[1].strip("{} ;").split(",")]
    return point_id, coords


def parse_line(line):
    parts = line.split("=")
    line_id = int(parts[0].split("(")[1].split(")")[0])
    point_ids = [int(p.strip()) for p in parts[1].strip("{} ;").split(",")]
    return line_id, point_ids


def remove_duplicate_lines(lines):
    return {line_id: point_ids for line_id, point_ids in lines.items() if point_ids[0] != point_ids[1]}


def find_line_loops(lines):
    line_loops = []
    visited_lines = set()

    for start_line_id in lines:
        if start_line_id in visited_lines:
            continue

        loop = [start_line_id]
        visited_lines.add(start_line_id)
        current_line_id = start_line_id

        while True:
            next_line_id = find_next_line(lines, current_line_id, visited_lines)
            if next_line_id is None:
                break

            loop.append(next_line_id)
            visited_lines.add(next_line_id)
            current_line_id = next_line_id

            if lines[current_line_id][1] == lines[start_line_id][0]:
                line_loops.append(loop)
                break

    return line_loops


def find_next_line(lines, current_line_id, visited_lines):
    current_end_point = lines[current_line_id][1]

    for line_id, point_ids in lines.items():
        if line_id not in visited_lines and point_ids[0] == current_end_point:
            return line_id

    return None


def remove_short_line_loops(line_loops, lines, points):
    valid_line_loops = []
    lines_to_remove = set()
    points_to_remove = set()

    for loop in line_loops:
        if len(loop) > 2:
            valid_line_loops.append(loop)
        else:
            for line_id in loop:
                lines_to_remove.add(line_id)
                points_to_remove.add(lines[line_id][0])
                points_to_remove.add(lines[line_id][1])

    lines = {line_id: point_ids for line_id, point_ids in lines.items() if line_id not in lines_to_remove}
    points = {point_id: coords for point_id, coords in points.items() if point_id not in points_to_remove}

    return valid_line_loops, lines, points


def generate_geo_content(points, lines, line_loops):
    geo_content = ""

    # 点の定義
    for point_id, coord in points.items():
        geo_content += f"Point({point_id}) = {{{coord[0]}, {coord[1]}, {coord[2]}}};\n"

    # 線の定義
    for line_id, point_ids in lines.items():
        geo_content += f"Line({line_id}) = {{{point_ids[0]}, {point_ids[1]}}};\n"

    # 線ループの定義
    for i, loop in enumerate(line_loops):
        geo_content += f"Line Loop({i + 1}) = {{{', '.join(map(str, loop))}}};\n"
        geo_content += f"Plane Surface({i + 1}) = {{{i + 1}}};\n"

    return geo_content
