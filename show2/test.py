import os
import shutil


def sort_and_rename_files(folder_path1,folder_path2):
    file_list = []  # 存儲所有.mp4檔案的列表
    
    # 遍歷資料夾，獲取所有.mp4檔案的路徑
    for file_name in os.listdir(folder_path1):
        if file_name.endswith(".mp4"):
            file_path = os.path.join(folder_path1, file_name)
            file_list.append(file_path)
    
    # 按文件的修改時間排序
    file_list.sort(key=os.path.getmtime)
    
    # 取得已存在的檔案數量
    existing_files_count = sum(1 for _ in os.listdir(folder_path1) if _.endswith(".mp4"))
    print("existing",existing_files_count)
    # 重新命名並移動檔案
    for file_name in os.listdir(folder_path2):
        print("file_name",file_name)
        now =os.path.join(folder_path2, file_name)
        print(now)
        if file_name.endswith(".mp4"):
            new_name = f"{existing_files_count + 1}.mp4"  # 新檔案名稱
            new_path = os.path.join(folder_path1, new_name)  # 新檔案路徑
            os.rename(now, new_path)
            print(f"Renamed: {now} -> {new_path}")

# 呼叫函式並指定要處理的資料夾路徑
folder_path1 = "E://things/master/DIH/DIH/show/01"
folder_path2 = "E://things/master/DIH/DIH/show/04"
sort_and_rename_files(folder_path1,folder_path2)
