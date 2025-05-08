import os
import math
import shutil

def normalize_vector(x, y):
    """ベクトルを正規化する"""
    magnitude = math.sqrt(x**2 + y**2)
    if magnitude == 0:
        return 0, 0
    return x / magnitude, y / magnitude

def rotate_vector_clockwise(x, y):
    """ベクトルを時計回りに90度回転させる"""
    return y, -x

def _process_geo_iteration(input_geo_path, output_geo_path, scale):
    """1回の反復で自己交差するLineのペアを探索と修正する内部関数"""
    try:
        with open(input_geo_path, 'r') as f:
            input_geo_content = f.read()
    except FileNotFoundError:
        print(f"エラー: 入力ファイル '{input_geo_path}' が見つかりません。")
        return False

    lines_data = []
    points = {}
    line_loops = {}
    plane_surfaces = {}
    max_point_index = 0
    line_to_loops = {}
    point_to_lines = {} # 各点を参照しているLineのリスト

    # 入力GEOファイルを解析
    for line_str in input_geo_content.strip().split('\n'):
        if line_str.startswith("Point("):
            parts = line_str[len("Point("):].split(")")
            if len(parts) > 1:
                try:
                    index = int(parts[0].strip())
                    coords_str = parts[1].split("=")[1].strip(' {};')
                    coords = [float(c.strip()) for c in coords_str.split(',')]
                    points[index] = coords
                    max_point_index = max(max_point_index, index)
                except ValueError as e:
                    print(f"ValueError (Point): {e}")
                    raise
                except Exception as e:
                    print(f"予期しないエラー (Point) が発生しました: {e}")
                    raise
            else:
                print("警告 (Point): 行の形式が不正です")
        elif line_str.startswith("Line("):
            parts = line_str[len("Line("):].split(")")
            if len(parts) > 1:
                try:
                    index = int(parts[0].strip())
                    point_indices_str = parts[1].split("=")[1].strip(' {};')
                    point_indices = tuple(int(p.strip()) for p in point_indices_str.split(','))
                    lines_data.append({'index': index, 'points': point_indices})
                    for point_index in point_indices:
                        if point_index not in point_to_lines:
                            point_to_lines[point_index] = []
                        point_to_lines[point_index].append(index)
                except ValueError as e:
                    print(f"ValueError (Line): {e}")
                    raise
                except Exception as e:
                    print(f"予期しないエラー (Line) が発生しました: {e}")
                    raise
            else:
                print("警告 (Line): 行の形式が不正です")
        elif line_str.startswith("Line Loop("):
            parts = line_str[len("Line Loop("):].split(")")
            if len(parts) > 1:
                try:
                    index = int(parts[0].strip())
                    line_indices_str = parts[1].split("=")[1].strip(' {};')
                    line_indices = [int(l.strip()) for l in line_indices_str.split(',')]
                    line_loops[index] = line_indices
                    for line_index in line_indices:
                        if line_index not in line_to_loops:
                            line_to_loops[line_index] = []
                        line_to_loops[line_index].append(index)
                except ValueError as e:
                    print(f"ValueError (Line Loop): {e}")
                    raise
                except Exception as e:
                    print(f"予期しないエラー (Line Loop) が発生しました: {e}")
                    raise
            else:
                print("警告 (Line Loop): 行の形式が不正です")
        elif line_str.startswith("Plane Surface("):
            parts = line_str[len("Plane Surface("):].split(")")
            if len(parts) > 1:
                try:
                    index = int(parts[0].strip())
                    loop_indices_str = parts[1].split("=")[1].strip(' {};')
                    loop_indices = [int(l.strip()) for l in loop_indices_str.split(',')]
                    plane_surfaces[index] = loop_indices
                except ValueError as e:
                    print(f"ValueError (Plane Surface): {e}")
                    raise
                except Exception as e:
                    print(f"予期しないエラー (Plane Surface) が発生しました: {e}")
                    raise
            else:
                print("警告 (Plane Surface): 行の形式が不正です")

    new_points = points.copy()
    updated_lines = list(lines_data)
    found_and_fixed = False
    next_point_index = max_point_index + 1
    lines_to_remove = set()
    points_to_remove = set()

    i = 0
    while i < len(updated_lines):
        line_l = updated_lines[i]
        j = i + 1
        while j < len(updated_lines):
            line_m = updated_lines[j]
            if set(line_l['points']) == set(line_m['points']) and line_l['points'] != line_m['points']:
                print(f"デバッグ (探索): i={i}, j={j} 自己交差の可能性: Line {line_l['index']} ({line_l['points']}) と Line {line_m['index']} ({line_m['points']})")

                is_adjacent = False
                loops_l = line_to_loops.get(line_l['index'], [])
                loops_m = line_to_loops.get(line_m['index'], [])

                for loop_index in set(loops_l) & set(loops_m):
                    loop_lines = line_loops.get(loop_index, [])
                    try:
                        index_l_in_loop = loop_lines.index(line_l['index'])
                        index_m_in_loop = loop_lines.index(line_m['index'])
                        print(f"デバッグ (隣接判定): Loop {loop_index}, Lines: {loop_lines}, Index l: {index_l_in_loop}, Index m: {index_m_in_loop}")
                        if abs(index_l_in_loop - index_m_in_loop) == 1 or \
                           (index_l_in_loop == 0 and index_m_in_loop == len(loop_lines) - 1) or \
                           (index_m_in_loop == 0 and index_l_in_loop == len(loop_lines) - 1):
                            is_adjacent = True
                            print(f"デバッグ (隣接判定): Line {line_l['index']} と Line {line_m['index']} は隣接しています。")
                            break
                    except ValueError:
                        continue

                if is_adjacent:
                    print(f"デバッグ (修正 - 隣接): Line {line_l['index']} ({line_l['points']}) と Line {line_m['index']} ({line_m['points']}) を削除します。")
                    lines_to_remove.add(line_l['index'])
                    lines_to_remove.add(line_m['index'])
                    # 隣接するLineが共有する点を削除対象に追加 (ただし、他のLineが参照していない場合のみ)
                    shared_points = set(line_l['points']) & set(line_m['points'])
                    for point_index in shared_points:
                        referring_lines = [
                            idx for idx in point_to_lines.get(point_index, [])
                            if idx not in lines_to_remove
                        ]
                        if not referring_lines:
                            print(f"デバッグ (削除対象 - 隣接共有点): Point {point_index} は他のLineから参照されていないため、削除対象に追加します。")
                            points_to_remove.add(point_index)
                        else:
                            print(f"デバッグ (削除対象 - 隣接共有点): Point {point_index} は Line(s) {referring_lines} から参照されているため、削除対象から除外します。")

                    # 削除されたLineが含まれるLine Loopを更新
                    lines_to_remove_from_loops = {}

                    for loop_index, loop_lines in line_loops.items():
                        lines_to_remove_current_loop = []
                        if line_l['index'] in loop_lines:
                            print(f"デバッグ (Line Loop検出 - Line L): Loop {loop_index}, Line {line_l['index']} を削除対象に追加.")
                            lines_to_remove_current_loop.append(line_l['index'])
                        if line_m['index'] in loop_lines:
                            print(f"デバッグ (Line Loop検出 - Line M): Loop {loop_index}, Line {line_m['index']} を削除対象に追加.")
                            lines_to_remove_current_loop.append(line_m['index'])

                        if lines_to_remove_current_loop:
                            lines_to_remove_from_loops[loop_index] = lines_to_remove_current_loop

                    # ループ終了後にまとめて Line Loop を更新
                    for loop_index_to_update, lines_to_remove in lines_to_remove_from_loops.items():
                        if loop_index_to_update in line_loops:
                            original_length = len(line_loops[loop_index_to_update])
                            line_loops[loop_index_to_update] = [
                                line_index for line_index in line_loops[loop_index_to_update]
                                if line_index not in lines_to_remove
                            ]
                            if len(line_loops[loop_index_to_update]) < original_length:
                                print(f"デバッグ (Line Loop更新): Loop {loop_index_to_update} から Line(s) {lines_to_remove} を削除しました。")

                    print("デバッグ (Line Loop更新処理完了): line_loops =", line_loops.get(2))
                    
                    found_and_fixed = True
                    print(f"デバッグ (削除対象): lines_to_remove = {lines_to_remove}, points_to_remove = {points_to_remove}")
                    break # 内側のループを抜ける
                else:
                    point_a_l, point_b_l = line_l['points']
                    a_coords = new_points[point_a_l]
                    b_coords = new_points[point_b_l]
                    ab_x_l = b_coords[0] - a_coords[0]
                    ab_y_l = b_coords[1] - a_coords[1]
                    line_length_l = math.sqrt(ab_x_l**2 + ab_y_l**2)
                    normal_x, normal_y = normalize_vector(*rotate_vector_clockwise(ab_x_l, ab_y_l))
                    offset = line_length_l * scale
                    c_coords = [a_coords[0] + normal_x * offset, a_coords[1] + normal_y * offset, 0.0]
                    d_coords = [b_coords[0] + normal_x * offset, b_coords[1] + normal_y * offset, 0.0]
                    print(f"デバッグ (修正 - 非隣接): Line {line_l['index']} ({line_l['points']}) と Line {line_m['index']} ({line_m['points']}) を修正します。")
                    print(f"デバッグ (修正 - 非隣接): 新しい点C({next_point_index}): {c_coords}, 新しい点D({next_point_index+1}): {d_coords}")
                    new_points[next_point_index] = c_coords
                    point_c = next_point_index
                    next_point_index += 1
                    new_points[next_point_index] = d_coords
                    point_d = next_point_index

                    # Line M の節点を置換
                    line_index_m = line_m['index']
                    if 1 <= line_index_m <= len(lines_data):
                        old_points_m = lines_data[line_index_m - 1]['points']
                        lines_data[line_index_m - 1]['points'] = (point_d, point_c)
                        print(f"デバッグ (修正 - 非隣接): Line {line_index_m} の節点を ({point_d}, {point_c}) に置換しました (旧: {old_points_m})。")
                    else:
                        print(f"エラー: Line index {line_index_m} は lines_data の範囲外です。")
                        continue

                    # Line Loop 内で Line M に隣接する Line P と Q を特定し、節点を置換
                    loop_index_m = None
                    for loop_idx, lines_in_loop in line_loops.items():
                        if line_index_m in lines_in_loop:
                            loop_index_m = loop_idx
                            break

                    if loop_index_m is not None:
                        loop_lines_m = line_loops[loop_index_m]
                        index_m_in_loop = loop_lines_m.index(line_index_m)
                        len_loop = len(loop_lines_m)

                        # 隣接する Line P (M の一つ前の Line)
                        index_p_in_loop = (index_m_in_loop - 1 + len_loop) % len_loop
                        line_index_p = loop_lines_m[index_p_in_loop]
                        if 1 <= line_index_p <= len(lines_data):
                            old_points_p = lines_data[line_index_p - 1]['points']
                            point_to_replace_p = old_points_p[1] if old_points_m[0] == lines_data[line_index_p - 1]['points'][1] else old_points_p[0]
                            new_points_p = (old_points_p[0] if point_to_replace_p == old_points_p[1] else point_d,
                                            old_points_p[1] if point_to_replace_p == old_points_p[0] else point_d)
                            lines_data[line_index_p - 1]['points'] = new_points_p
                            print(f"デバッグ (修正 - 隣接P): Line {line_index_p} の節点を ({new_points_p}) に置換しました (旧: {old_points_p})。")
                        else:
                            print(f"エラー: Line index {line_index_p} は lines_data の範囲外です。")

                        # 隣接する Line Q (M の一つ後の Line)
                        index_q_in_loop = (index_m_in_loop + 1) % len_loop
                        line_index_q = loop_lines_m[index_q_in_loop]
                        if 1 <= line_index_q <= len(lines_data):
                            old_points_q = lines_data[line_index_q - 1]['points']
                            point_to_replace_q = old_points_q[0] if old_points_m[1] == lines_data[line_index_q - 1]['points'][0] else old_points_q[1]
                            new_points_q = (old_points_q[0] if point_to_replace_q == old_points_q[1] else point_c,
                                            old_points_q[1] if point_to_replace_q == old_points_q[0] else point_c)
                            lines_data[line_index_q - 1]['points'] = new_points_q
                            print(f"デバッグ (修正 - 隣接Q): Line {line_index_q} の節点を ({new_points_q}) に置換しました (旧: {old_points_q})。")
                        else:
                            print(f"エラー: Line index {line_index_q} は lines_data の範囲外です。")
                    else:
                        print(f"警告: Line {line_index_m} がどの Line Loop にも属していません。隣接するLineの置換はスキップします。")                    


                    
                    found_and_fixed = True
                    break # 内側のループを抜ける
            j += 1
        if found_and_fixed:
            print("デバッグ (更新): updated_lines をフィルタリング (削除対象:", lines_to_remove, ")")
            updated_lines = [line for line in updated_lines if line['index'] not in lines_to_remove]
            print("デバッグ (更新): updated_lines =", [l['index'] for l in updated_lines])
            lines_to_remove.clear()
            break # 外側のループを抜ける
        i += 1

    # 最終的な出力
    output_lines = []
    for index, coords in sorted(new_points.items()):
        if index not in points_to_remove:
            output_lines.append(f"Point({index}) = {{{coords[0]}, {coords[1]}, {coords[2]}}};")

    final_lines = [line for line in updated_lines if line['index'] not in lines_to_remove]
    print("デバッグ (出力準備): final_lines =", [l['index'] for l in final_lines])
    print("デバッグ (出力準備 - 直前): line_loops =", line_loops.get(2)) # 出力準備直前の
    print("デバッグ (出力準備): plane_surfaces =", plane_surfaces)

    for line in sorted(final_lines, key=lambda x: x['index']):
        valid_points = tuple(p for p in line['points'] if p not in points_to_remove)
        if len(valid_points) == 2:
            output_lines.append(f"Line({line['index']}) = {{{valid_points[0]}, {valid_points[1]}}};")

    for index, loop_lines in sorted(line_loops.items()):
        if loop_lines:  # 空のLine Loopは出力しない
            output_lines.append(f"Line Loop({index}) = {{{', '.join(map(str, loop_lines))}}};")

    for index, surface_loops in sorted(plane_surfaces.items()):
        output_lines.append(f"Plane Surface({index}) = {{{', '.join(map(str, surface_loops))}}};")

    output_geo_content = "\n".join(output_lines)

    # 新しいGEOファイルを出力
    try:
        os.makedirs(os.path.dirname(output_geo_path), exist_ok=True)
        with open(output_geo_path, 'w') as f:
            f.write(output_geo_content)
        print(f"デバッグ (_process): 終了時の found_and_fixed: {found_and_fixed}")
        return found_and_fixed
    except Exception as e:
        print(f"エラー: 出力ファイルへの書き込みに失敗しました: {e}")
        print(f"デバッグ (_process): 終了時の found_and_fixed (エラー発生): False")
        return False

def fix_self_intersecting_lines(input_path, output_path, scale=0.1, max_iterations=10):
    current_input_path = input_path
    intermediate_output_path = output_path.replace(".geo", "_intermediate_1.geo")
    final_intermediate_path = input_path
    last_modified_intermediate_path = None
    iteration = 0
    found_and_fixed_in_iteration = True

    while iteration < max_iterations and found_and_fixed_in_iteration:
        iteration += 1
        print(f"反復 {iteration}: 自己交差するLineのペアを探索と修正. 入力: '{current_input_path}', 結果は '{intermediate_output_path}' に保存.")
        found_and_fixed = _process_geo_iteration(current_input_path, intermediate_output_path, scale)
        print(f"デバッグ: 反復 {iteration} の _process_geo_iteration の戻り値: {found_and_fixed}")

        if found_and_fixed:
            final_intermediate_path = intermediate_output_path
            last_modified_intermediate_path = intermediate_output_path
            current_input_path = intermediate_output_path
            intermediate_output_path = output_path.replace(".geo", f"_intermediate_{iteration + 1}.geo")
            found_and_fixed_in_iteration = True
        else:
            found_and_fixed_in_iteration = False
            if iteration >= 2:
                print(f"デバッグ: 自己交差が見つからない状態が2回続いたため、反復を終了します。final_intermediate_path は '{final_intermediate_path}' です。")
                break
            else:
                print(f"デバッグ: 反復 {iteration} で修正なし。次の反復に進みます。")
                current_input_path = input_path

    print(f"デバッグ: 反復終了後の final_intermediate_path: {final_intermediate_path}")
    print(f"デバッグ: 反復終了後の last_modified_intermediate_path: {last_modified_intermediate_path}")

    if last_modified_intermediate_path:
        print(f"デバッグ: コピー元ファイル (修正あり): {last_modified_intermediate_path}")
        shutil.copy2(last_modified_intermediate_path, output_path)
    else:
        print(f"デバッグ: コピー元ファイル (fallback): {input_path}")
        shutil.copy2(input_path, output_path)

    """    
    # 中間ファイルの削除
    for i in range(1, iteration + 1):
        intermediate_file = output_path.replace(".geo", f"_intermediate_{i}.geo")
        if os.path.exists(intermediate_file):
            os.remove(intermediate_file)
            print(f"中間ファイル '{intermediate_file}' を削除.")
    """
