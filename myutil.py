import os
import shutil
import pandas as pd

def delete_and_recreate_directory(directory_path):
    """
    指定されたディレクトリが存在する場合、ユーザーに削除の許可を求め、
    許可された場合にディレクトリを削除し、同名のディレクトリを再作成する関数。

    Args:
        directory_path (str): ディレクトリのパス。
    """

    if os.path.exists(directory_path):
        print(f"ディレクトリ '{directory_path}' は既に存在します。")
        response = input(f"ディレクトリ '{directory_path}' を削除して再作成しますか？ (y/n): ").lower()
        if response == 'y':
            try:
                shutil.rmtree(directory_path)  # ディレクトリを削除
                os.makedirs(directory_path)  # ディレクトリを再作成
                print(f"ディレクトリ '{directory_path}' を削除し、再作成しました。")
            except Exception as e:
                print(f"エラー: ディレクトリの削除または再作成に失敗しました。{e}")
        else:
            print("処理を中断します。")
    else:
        try:
            os.makedirs(directory_path)  # ディレクトリを作成
            print(f"ディレクトリ '{directory_path}' を作成しました。")
        except Exception as e:
            print(f"エラー: ディレクトリの作成に失敗しました。{e}")

def check_file_and_confirm(filepath):
    """
    ファイルが存在するかどうかを調べ、存在する場合に処理を継続するかどうかをユーザーに確認する関数。

    Args:
        filepath (str): ファイルパス。

    Returns:
        bool: 処理を継続する場合はTrue、中断する場合はFalse。
    """

    if os.path.exists(filepath):
        print(f"ファイル '{filepath}' は既に存在します。")
        response = input("処理を継続しますか？ (y/n): ").lower()
        if response == 'y':
            return True
        else:
            print("処理を中断します。")
            return False
    else:
        return True  # ファイルが存在しない場合は処理を継続


def nlopt_result_message(opt):
    # 結果コード
    result_code = opt.last_optimize_result()

    # 結果コードに対応するメッセージを生成
    if result_code == 3:
            result_message = "目的関数の相対許容誤差に達しました。(code 3)"
    elif result_code == 4:
            result_message = "最適化変数の相対許容誤差に達しました。(code 4)"
    elif result_code == 5:
            result_message = "最大評価回数に達しました。(code 5)"
    elif result_code == 6:
            result_message = "最大実行時間に達しました。(code 6)"
    elif result_code == 1:
            result_message = "最適化が成功しました。(code 1)"
    else:
            result_message = opt.get_errmsg()
    return result_message


def save_to_space_separated_text(call_counts, f_values, x_values, opt_dir, filename="history.txt"):
    """
    x_values の各要素が z_dim 次元である場合に、スペース区切りのテキストファイルに保存する。

    Args:
        call_counts (list): call_counts のリスト。
        f_values (list): f_values のリスト。
        x_values (list): x_values のリスト (各要素は z_dim 次元のリスト)。
        opt_dir (str): 保存先のディレクトリ。
        filename (str): 保存するファイル名。
    """

    z_dim = len(x_values[0])  # x_values の要素の次元を取得

    # DataFrame を作成
    data = {"call_counts": call_counts, "f_values": f_values}

    # x_values の各要素を列に展開
    for i in range(z_dim):
        data[f"x_values_col{i}"] = [x[i] for x in x_values]

    df = pd.DataFrame(data)

    # スペース区切りのテキストファイルに保存
    df.to_csv(os.path.join(opt_dir, filename), sep=' ', index=False, header=False)
