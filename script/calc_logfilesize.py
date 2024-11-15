import os
import re
from collections import defaultdict

def format_size(size):
    """サイズをKB、MB、GBに変換して適切な単位で表示"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024

def calculate_specific_log_size(directory_path, condition, match_type="exact", show_extension_summary=False, delete_files=False):
    total_size = 0
    matching_files = []
    total_size_by_extension = defaultdict(int)

    # match_typeのチェック
    assert match_type in ["exact", "extension", "regex"], f"Invalid match_type: {match_type}"

    # 指定したディレクトリ内を再帰的に探索
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if show_extension_summary:
                # フルパスとファイルサイズを取得
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)
                # 拡張子ごとのサイズの辞書に追加
                extension = os.path.splitext(file)[1]
                total_size_by_extension[extension] += file_size

            # マッチタイプに応じて条件を適用
            if match_type == "exact" and file == condition:
                match = True
            elif match_type == "extension" and file.endswith(condition):
                match = True
            elif match_type == "regex" and re.fullmatch(condition, file):
                match = True
            else:
                match = False
            if match:
                # フルパスとファイルサイズを取得
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)
                total_size += file_size
                matching_files.append((file_path, file_size))
    
    # # 各ログファイルのフルパスとサイズを表示 (見つかったファイル数も表示)
    # print(f"Files named '{condition}':")
    # for file_path, file_size in matching_files:
    #     print(f"{file_path} - {format_size(file_size)}")

    # 合計サイズを適切な単位で表示
    print(f"\nTotal size of files named '{condition}': {format_size(total_size)}")
    print(f"Number of matched log files: {len(matching_files)}")

    if show_extension_summary:
        # 拡張子ごとのサイズを表示
        # 表示する際は，サイズの大きい順にソート
        print("\nSize summary by extension:")
        for extension, size in sorted(total_size_by_extension.items(), key=lambda x: x[1], reverse=True):
            print(f"{extension}: {format_size(size)}")
    
    # ファイルを削除
    if delete_files:
        for file_path, file_size in matching_files:
            try:
                os.remove(file_path)
                print(f"Deleted: {file_path} - {format_size(file_size)}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")


if __name__ == "__main__":
    directory_path = "/src"  # 対象のディレクトリを指定
    log_filename = "101_change_localized_weights_run_all.log"  # 対象のファイル名を指定
    calculate_specific_log_size(directory_path, condition=log_filename, match_type="exact")
    # calculate_specific_log_size(directory_path, condition=".log", match_type="extension", show_extension_summary=True)