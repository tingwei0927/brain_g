import os
import shutil


#移動mp4           
def sort_and_rename_files(folder_path1,target_kill):#目標資料夾跟要清空的資料夾
    file_list = []  # 存儲folder_path1所有.mp4檔案的列表
    
    # 遍歷資料夾，獲取所有.mp4檔案的路徑
    for file_name in os.listdir(folder_path1):
        if file_name.endswith(".mp4"):
            file_path = os.path.join(folder_path1, file_name)
            file_list.append(file_path)
    
    # 按文件的修改時間排序
    file_list.sort(key=os.path.getmtime)
    
    # 取得已存在的檔案數量
    existing_files_count = sum(1 for _ in os.listdir(folder_path1) if _.endswith(".mp4"))
    #print("existing",existing_files_count)
    # 重新命名並移動檔案

    if target_kill.endswith(".mp4"):
        # 關閉檔案
        file_to_rename = open(target_kill, "r")
        file_to_rename.close()
        new_name = f"{existing_files_count + 1}.mp4"  # 新檔案名稱
        new_path = os.path.join(folder_path1, new_name)  # 新檔案路徑
        os.rename(target_kill, new_path)
        #print(f"Renamed: {target_kill} -> {new_path}")



#這是用來刪除資料夾的
def removefile(filepath):
    # 欲刪除的資料夾路徑
    folder_path = filepath  #"./here"

    # 刪除資料夾及其內容
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)#這是用來刪除資料夾的
        #print(f"資料夾 {folder_path} 及其內容已刪除")


