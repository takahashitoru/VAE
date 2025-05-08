import sys
import re

def remove_overlapping_lines(input_geo_file, output_geo_file):
    """
    入力の.geoファイルを読み込み、始点と終点が逆になっている隣接するLineを
    Line Loop内で検出し、それらのLineと共通の点（終点であり次のLineの始点）を
    削除した内容を新しい.geoファイルに書き出します。
    入力ファイルの解析ロジックは resolve_self_intersection() に準拠します。
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

    # 入力GEOファイルを解析
    for line_str in input_geo_content.strip().split('\n'):
        line_str = line_str.strip()
        if line_str.startswith("Point("):
            match = re.match(r"Point\((\d+)\)\s*=\s*\{(.*?)\};", line_str)
            if match:
                try:
                    index = int(match.group(1))
                    coords_str = match.group(2)
                    coords = [float(c.strip()) for c in coords_str.split(',')]
                    points[index] = coords[:3]
                except ValueError as e:
                    print(f"ValueError (Point): {e}")
                except Exception as e:
                    print(f"予期しないエラー (Point): {e}")
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
                except ValueError as e:
                    print(f"ValueError (Line): {e}")
                except Exception as e:
                    print(f"予期しないエラー (Line): {e}")
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
                    print(f"ValueError (Line Loop): {e}")
                except Exception as e:
                    print(f"予期しないエラー (Line Loop): {e}")
        elif line_str.startswith("Plane Surface("):
            match = re.match(r"Plane Surface\((\d+)\)\s*=\s*\{(.*?)\};", line_str)
            if match:
                try:
                    index = int(match.group(1))
                    loops_str = match.group(2)
                    loop_indices = [int(l.strip()) for l in loops_str.split(',')]
                    plane_surfaces[index] = loop_indices
                except ValueError as e:
                    print(f"ValueError (Plane Surface): {e}")
                except Exception as e:
                    print(f"予期しないエラー (Plane Surface): {e}")

    lines_to_remove = set()
    points_to_remove = set()
    updated_line_loops = {}
    lines_dict = {line['index']: line['points'] for line in lines}

    for loop_index, loop_lines in line_loops.items():
        n_lines = len(loop_lines)
        updated_loop_lines = []
        removed_in_loop = set()

        for i in range(n_lines):
            current_line_index = loop_lines[i]
            if current_line_index in removed_in_loop:
                continue

            next_index = (i + 1) % n_lines
            next_line_index = loop_lines[next_index]

            if current_line_index in lines_dict and next_line_index in lines_dict:
                line_current = lines_dict[current_line_index]
                line_next = lines_dict[next_line_index]

                if tuple(sorted(line_current)) == tuple(sorted(line_next)) and line_current != line_next:
                    print(f"デバッグ: 自己交差 Line ペア L({current_line_index}) = {line_current}, M({next_line_index}) = {line_next} を検出 (Loop {loop_index})")
                    lines_to_remove.add(current_line_index)
                    lines_to_remove.add(next_line_index)
                    removed_in_loop.add(current_line_index)
                    removed_in_loop.add(next_line_index)

                    # 共通の点（current_lineの終点、next_lineの始点）を削除候補に追加
                    points_to_remove.add(line_current[1])
                else:
                    updated_loop_lines.append(current_line_index)
            else:
                updated_loop_lines.append(current_line_index)

        updated_line_loops[loop_index] = [line_index for line_index in loop_lines if line_index not in lines_to_remove]

    print("デバッグ: 削除対象の Line インデックス:", lines_to_remove)
    print("デバッグ: 削除対象の Point インデックス:", points_to_remove)
    #print("デバッグ: 更新された line_loops:", updated_line_loops)

    # 新しい GEO ファイルの内容を生成
    output_geo_content = ""

    # Point を出力 (削除されなかった Point のみ)
    for index, coords in sorted(points.items()):
        if index not in points_to_remove:
            output_geo_content += f"Point({index}) = {{{coords[0]}, {coords[1]}, {coords[2]}}};\n"

    # Line を出力 (削除されなかった Line のみ)
    for line in sorted(lines, key=lambda x: x['index']):
        if line['index'] not in lines_to_remove and line['points'][0] not in points_to_remove and line['points'][1] not in points_to_remove:
            output_geo_content += f"Line({line['index']}) = {{{line['points'][0]}, {line['points'][1]}}};\n"

    # Line Loop を出力 (更新された Line インデックスを使用)
    for index, loop_lines in sorted(line_loops.items()):
        updated_loop_lines_for_output = [line_index for line_index in loop_lines if line_index not in lines_to_remove]
        if updated_loop_lines_for_output:
            output_geo_content += f"Line Loop({index}) = {{{', '.join(map(str, updated_loop_lines_for_output))}}};\n"
        # Line Loopが空になった場合は出力しない

    # Plane Surface を出力 (元の参照関係を維持)
    for index, surface_loops in sorted(plane_surfaces.items()):
        output_geo_content += f"Plane Surface({index}) = {{{', '.join(map(str, surface_loops))}}};\n"

    try:
        with open(output_geo_file, 'w') as f:
            f.write(output_geo_content)
        print(f"処理完了: 自己交差するLineペアとその共通点を削除し、'{output_geo_file}' に出力しました。")
    except Exception as e:
        print(f"エラー: 出力ファイル '{output_geo_file}' への書き込みに失敗しました: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("使用方法: python script.py <入力.geo> <出力.geo>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_geo_file = sys.argv[2]
    remove_overlapping_lines(input_file, output_geo_file)
