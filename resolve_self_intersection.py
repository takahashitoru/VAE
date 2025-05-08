import os

def resolve_self_intersection(input_geo_file, output_geo_file):
    """
    入力の.geoファイルを読み込み、自己交差しているLine Loopを検出し、
    指定された方法で解消した内容を新しい.geoファイルに書き出します。
    自己交差がない場合は、入力ファイルの内容をそのまま出力します。
    Line LoopやPlane Surfaceの定義は変更しません。
    提供された解析ロジックを使用します。
    """

    try:
        with open(input_geo_file, 'r') as f:
            input_geo_content = f.read()
    except FileNotFoundError:
        print(f"エラー: 入力ファイル '{input_geo_file}' が見つかりません。")
        return

    lines = []
    points = {}
    line_loops = {}
    plane_surfaces = {}
    max_point_index = 0
    max_line_index = 0
    original_geo_lines = input_geo_content.strip().split('\n')

    # 入力GEOファイルを解析
    for line_str in original_geo_lines:
        line_str = line_str.strip()
        if line_str.startswith("Point("):
            parts = line_str[len("Point("):].split(")")
            if len(parts) > 1:
                try:
                    index_str = parts[0].strip()
                    index = int(index_str)
                    coords_part = parts[1].strip()
                    if coords_part.startswith("=") and coords_part[coords_part.find("=") + 1:].lstrip().startswith("{"):
                        coords_str = coords_part[coords_part.find("=") + 1:].lstrip().strip('{}').rstrip(';')
                        coords_list_str = coords_str.split(',')
                        coords = [float(c.strip().rstrip('}')) for c in coords_list_str]
                        points[index] = coords
                        max_point_index = max(max_point_index, index)
                except ValueError as e:
                    print(f"ValueError (Point) が発生しました: {e}")
                    raise
                except Exception as e:
                    print(f"予期しないエラー (Point) が発生しました: {e}")
                    raise
        elif line_str.startswith("Line("):
            parts = line_str[len("Line("):].split(")")
            if len(parts) > 1:
                try:
                    index_str = parts[0].strip()
                    index = int(index_str)
                    points_part = parts[1].strip()
                    if points_part.startswith("=") and points_part[points_part.find("=") + 1:].lstrip().startswith("{"):
                        point_indices_str = points_part[points_part.find("=") + 1:].lstrip().strip('{}').split(',')
                        point_indices = tuple(int(p.strip().rstrip(';}')) for p in point_indices_str)
                        lines.append({'index': index, 'points': point_indices, 'original_str': line_str})
                        max_line_index = max(max_line_index, index)
                except ValueError as e:
                    print(f"ValueError (Line) が発生しました: {e}")
                    raise
                except Exception as e:
                    print(f"予期しないエラー (Line) が発生しました: {e}")
                    raise
        elif line_str.startswith("Line Loop("):
            parts = line_str[len("Line Loop("):].split(")")
            if len(parts) > 1:
                try:
                    index_str = parts[0].strip()
                    index = int(index_str)
                    lines_part = parts[1].strip()
                    if lines_part.startswith("=") and lines_part[lines_part.find("=") + 1:].lstrip().startswith("{"):
                        line_indices_str = lines_part[lines_part.find("=") + 1:].lstrip().strip('{}').split(',')
                        line_indices = [int(l.strip().rstrip(';}')) for l in line_indices_str]
                        line_loops[index] = line_indices
                except ValueError as e:
                    print(f"ValueError (Line Loop) が発生しました: {e}")
                    raise
                except Exception as e:
                    print(f"予期しないエラー (Line Loop) が発生しました: {e}")
                    raise
        elif line_str.startswith("Plane Surface("):
            parts = line_str[len("Plane Surface("):].split(")")
            if len(parts) > 1:
                try:
                    index_str = parts[0].strip()
                    index = int(index_str)
                    loops_part = parts[1].strip()
                    if loops_part.startswith("=") and loops_part[loops_part.find("=") + 1:].lstrip().startswith("{"):
                        loop_indices_str = loops_part[loops_part.find("=") + 1:].lstrip().strip('{}').split(',')
                        loop_indices = [int(l.strip().rstrip(';}')) for l in loop_indices_str]
                        plane_surfaces[index] = loop_indices
                except ValueError as e:
                    print(f"ValueError (Plane Surface) が発生しました: {e}")
                    raise
                except Exception as e:
                    print(f"予期しないエラー (Plane Surface) が発生しました: {e}")
                    raise

    intersecting_points_found = True
    current_geo_content = input_geo_content
    iteration = 0

    while intersecting_points_found:
        iteration += 1
        print(f"--- 自己交差解消イテレーション {iteration} ---")

        lines = []
        points = {}
        line_loops = {}
        plane_surfaces = {}
        max_point_index = 0
        max_line_index = 0
        parsed_lines = current_geo_content.strip().split('\n')

        # 再度解析
        for line_str in parsed_lines:
            line_str = line_str.strip()
            if line_str.startswith("Point("):
                parts = line_str[len("Point("):].split(")")
                if len(parts) > 1:
                    try:
                        index_str = parts[0].strip()
                        index = int(index_str)
                        coords_part = parts[1].strip()
                        if coords_part.startswith("=") and coords_part[coords_part.find("=") + 1:].lstrip().startswith("{"):
                            coords_str = coords_part[coords_part.find("=") + 1:].lstrip().strip('{}').rstrip(';')
                            coords_list_str = coords_str.split(',')
                            coords = [float(c.strip().rstrip('}')) for c in coords_list_str]
                            points[index] = coords
                            max_point_index = max(max_point_index, index)
                    except ValueError as e:
                        print(f"ValueError (Point) が発生しました: {e}")
                        raise
                    except Exception as e:
                        print(f"予期しないエラー (Point) が発生しました: {e}")
                        raise
            elif line_str.startswith("Line("):
                parts = line_str[len("Line("):].split(")")
                if len(parts) > 1:
                    try:
                        index_str = parts[0].strip()
                        index = int(index_str)
                        points_part = parts[1].strip()
                        if points_part.startswith("=") and points_part[points_part.find("=") + 1:].lstrip().startswith("{"):
                            point_indices_str = points_part[points_part.find("=") + 1:].lstrip().strip('{}').split(',')
                            point_indices = tuple(int(p.strip().rstrip(';}')) for p in point_indices_str)
                            lines.append({'index': index, 'points': point_indices})
                            max_line_index = max(max_line_index, index)
                    except ValueError as e:
                        print(f"ValueError (Line) が発生しました: {e}")
                        raise
                    except Exception as e:
                        print(f"予期しないエラー (Line) が発生しました: {e}")
                        raise
            elif line_str.startswith("Line Loop("):
                parts = line_str[len("Line Loop("):].split(")")
                if len(parts) > 1:
                    try:
                        index_str = parts[0].strip()
                        index = int(index_str)
                        lines_part = parts[1].strip()
                        if lines_part.startswith("=") and lines_part[lines_part.find("=") + 1:].lstrip().startswith("{"):
                            line_indices_str = lines_part[lines_part.find("=") + 1:].lstrip().strip('{}').split(',')
                            line_indices = [int(l.strip().rstrip(';}')) for l in line_indices_str]
                            line_loops[index] = line_indices
                    except ValueError as e:
                        print(f"ValueError (Line Loop) が発生しました: {e}")
                        raise
                    except Exception as e:
                        print(f"予期しないエラー (Line Loop) が発生しました: {e}")
                        raise
            elif line_str.startswith("Plane Surface("):
                parts = line_str[len("Plane Surface("):].split(")")
                if len(parts) > 1:
                    try:
                        index_str = parts[0].strip()
                        index = int(index_str)
                        loops_part = parts[1].strip()
                        if loops_part.startswith("=") and loops_part[loops_part.find("=") + 1:].lstrip().startswith("{"):
                            loop_indices_str = loops_part[loops_part.find("=") + 1:].lstrip().strip('{}').split(',')
                            loop_indices = [int(l.strip().rstrip(';}')) for l in loop_indices_str]
                            plane_surfaces[index] = loop_indices
                    except ValueError as e:
                        print(f"ValueError (Plane Surface) が発生しました: {e}")
                        raise
                    except Exception as e:
                        print(f"予期しないエラー (Plane Surface) が発生しました: {e}")
                        raise

        # 各点に接続するラインを解析
        point_connections = {}
        lines_data = {line['index']: line['points'] for line in lines}
        for line_id, (start, end) in lines_data.items():
            if start not in point_connections:
                point_connections[start] = []
            point_connections[start].append(line_id)
            if end not in point_connections:
                point_connections[end] = []
            point_connections[end].append(line_id)

        intersecting_points = [p for p, connected in point_connections.items() if len(connected) == 4]
        print(f"デバッグ(交差点候補): 4本のラインが接続する点:", intersecting_points)

        found_intersection_in_this_iteration = False
        modified_lines_this_iteration = {}
        new_points_this_iteration = {}
        next_max_point_index = max_point_index

        points_to_process = sorted(list(intersecting_points))
        processed_points = set()

        for b_point in points_to_process:
            if b_point in processed_points or b_point not in points:
                continue

            connected_lines_ids = sorted(point_connections.get(b_point, []))
            connected_lines = {line['index']: line['points'] for line in lines if line['index'] in connected_lines_ids}

            if len(connected_lines_ids) == 4:
                line_p_id = connected_lines_ids[0]
                line_q_id = connected_lines_ids[1]
                line_r_id = connected_lines_ids[2]
                line_s_id = connected_lines_ids[3]

                if all(l_id in lines_data for l_id in [line_p_id, line_q_id, line_r_id, line_s_id]):
                    start_p, end_p = lines_data[line_p_id]
                    start_q, end_q = lines_data[line_q_id]
                    start_r, end_r = lines_data[line_r_id]
                    start_s, end_s = lines_data[line_s_id]

                    if (start_p == b_point or end_p == b_point) and \
                       (start_q == b_point or end_q == b_point) and \
                       (start_r == b_point or end_r == b_point) and \
                       (start_s == b_point or end_s == b_point):
                        found_intersection_in_this_iteration = True
                        found_intersection = True
                        next_max_point_index += 1
                        new_point_index_1 = next_max_point_index
                        next_max_point_index += 1
                        new_point_index_2 = next_max_point_index

                        point_a_p = end_p if start_p == b_point else start_p
                        point_c_q = end_q if start_q == b_point else start_q
                        point_d_r = start_r if end_r == b_point else end_r
                        point_e_s = start_s if end_s == b_point else start_s

                        if all(p in points for p in [point_a_p, point_c_q, point_d_r, point_e_s]):
                            mid_ac = [(points[point_a_p][i] + points[point_c_q][i]) / 2 for i in range(len(points[point_a_p]))]
                            mid_de = [(points[point_d_r][i] + points[point_e_s][i]) / 2 for i in range(len(points[point_d_r]))]
                            new_points_this_iteration[new_point_index_1] = mid_ac
                            new_points_this_iteration[new_point_index_2] = mid_de

                            def update_line_points(start_node, end_node, old_point, new_point):
                                if start_node == old_point:
                                    start_node = new_point
                                if end_node == old_point:
                                    end_node = new_point
                                return start_node, end_node

                            modified_lines_this_iteration[line_p_id] = update_line_points(start_p, end_p, b_point, new_point_index_1)
                            modified_lines_this_iteration[line_q_id] = update_line_points(start_q, end_q, b_point, new_point_index_1)
                            modified_lines_this_iteration[line_r_id] = update_line_points(start_r, end_r, b_point, new_point_index_2)
                            modified_lines_this_iteration[line_s_id] = update_line_points(start_s, end_s, b_point, new_point_index_2)

                            processed_points.add(b_point)
                            break # 一つの交差点を解消したら、このイテレーションを終了し、中間ファイルを書き出す
                        else:
                            print(f"デバッグ(点B={b_point}): 接続点のいずれかが存在しません")
                    else:
                        print(f"デバッグ(点B={b_point}): 自己交差条件不一致")
                else:
                    print(f"デバッグ(点B={b_point}): 接続ラインIDに対応するラインデータが見つかりません")
            elif len(connected_lines_ids) > 0:
                print(f"デバッグ(点B={b_point}): 接続ライン数: {len(connected_lines_ids)}")

        # 中間的な geo ファイルの内容を生成
        new_geo_lines = []

        # Point 定義の追加 (元の点と新しい点)
        for point_id in sorted(points.keys()):
            if point_id not in processed_points:
                new_geo_lines.append(f'Point({point_id}) = {{{", ".join(map(str, points[point_id]))}}};')
        for point_id, coords in sorted(new_points_this_iteration.items()):
            new_geo_lines.append(f'Point({point_id}) = {{{", ".join(map(str, coords))}}};')

        # Line 定義の追加 (修正されたラインと元のライン)
        processed_line_ids = set()
        for line_data in sorted(lines, key=lambda x: x['index']):
            line_id = line_data['index']
            start, end = line_data['points']
            if line_id in modified_lines_this_iteration:
                new_geo_lines.append(f'Line({line_id}) = {{{modified_lines_this_iteration[line_id][0]}, {modified_lines_this_iteration[line_id][1]}}};')
                processed_line_ids.add(line_id)
            elif line_id not in modified_lines_this_iteration:
                new_geo_lines.append(f'Line({line_id}) = {{{start}, {end}}};')
                processed_line_ids.add(line_id)

        # Line Loop と Plane Surface はそのまま追加
        for index, loop_lines in sorted(line_loops.items()):
            new_geo_lines.append(f'Line Loop({index}) = {{{", ".join(map(str, loop_lines))}}};')
        for index, surface_loops in sorted(plane_surfaces.items()):
            new_geo_lines.append(f'Plane Surface({index}) = {{{", ".join(map(str, surface_loops))}}};')

        current_geo_content = '\n'.join(new_geo_lines)

        if found_intersection_in_this_iteration:
            # 中間ファイルを書き出す
            base, ext = os.path.splitext(output_geo_file)
            intermediate_output_file = f"{base}_intermediate_{iteration}{ext}"
            try:
                with open(intermediate_output_file, 'w') as f:
                    f.write(current_geo_content)
                print(f"自己交差を検出し、解消を試み、中間結果を '{intermediate_output_file}' に書き出しました。")
                input_geo_file = intermediate_output_file # 次のイテレーションの入力ファイルとして使用
            except Exception as e:
                print(f"エラー: 中間ファイルへの書き込みに失敗しました: {e}")
                break
        else:
            intersecting_points_found = False
            print("自己交差は検出されませんでした。")

    # 最終的な出力ファイルを書き出す
    try:
        with open(output_geo_file, 'w') as f:
            f.write(current_geo_content)
        if found_intersection:
            print(f"最終結果を '{output_geo_file}' に書き出しました。")
        else:
            print(f"自己交差は検出されなかったため、元の内容を '{output_geo_file}' に書き出しました。")
    except Exception as e:
        print(f"エラー: 最終的な出力ファイルへの書き込みに失敗しました: {e}")
