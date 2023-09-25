#負責一直讀六個資料夾裡面有沒有新的mp4檔，有的話就要呼叫相對應的程式去處理跟評分
import csv
import os
import shutil
import time
import for_video_010
import for_video_04
import for_video_1202
import for_video_1402
import for_video_08
import for_video_09
import for_score_01



def writetxt(content,path):
    
    file_path = path  # 檔案路徑
    print("text",content)
    # 要寫入的三個字串內容
    text_content = content
    new_list = [str(x) for x in text_content]

    # 開啟檔案並寫入文字內容
    with open(file_path, "w") as file:
        file.writelines("\n".join(new_list))

    print("Text has been written to the file.")


def delete_mp4_files(folder_path):
    # 取得資料夾中所有文件的列表
    file_list = os.listdir(folder_path)
    
    for file_name in file_list:
        # 檢查文件是否是以 ".mp4" 為後綴
        if file_name.endswith(".mp4"):
            # 构建完整的文件路徑
            file_path = os.path.join(folder_path, file_name)
            
            # 刪除文件
            os.remove(file_path)
            print(f"已刪除文件：{file_path}")


def delete_txt(folder_path):
    folder_path
    # 檢查文件是否是以 ".mp4" 為後綴
    if folder_path.endswith(".txt"):
        # 刪除文件
        os.remove(folder_path)
        print(f"已刪除文件：{folder_path}")

    

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
    # print("existing",existing_files_count)
    # 重新命名並移動檔案

    if target_kill.endswith(".mp4"):
        new_name = f"{existing_files_count + 1}.mp4"  # 新檔案名稱
        new_path = os.path.join(folder_path1, new_name)  # 新檔案路徑
        os.rename(target_kill, new_path)
        # print(f"Renamed: {target_kill} -> {new_path}")



#這是用來刪除資料夾的
def removefile(filepath):
    # 欲刪除的資料夾路徑
    folder_path = filepath  #"./here"

    # 刪除資料夾及其內容
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)#這是用來刪除資料夾的
        print(f"資料夾 {folder_path} 及其內容已刪除")

    else:
        print(f"找不到資料夾 {folder_path}")


video_num1 = []
video_num4 = []
video_num8 = []
video_num9 = []
video_num12 = []
video_num14 = []


def run_process(folderpath):
    global video_num1, video_num4, video_num8, video_num9, video_num12, video_num14

    # 要处理的文件夹路径

    # 获取文件夹中所有文件的列表
    video = folderpath

    folder_path = "E://things/master/DIH/DIH/show2/" + video 
    print("folder_path",folder_path)
    
    
    print()
    file_list = os.listdir(folder_path)# 遍历文件列表
    print(file_list)
    for file_name in file_list:  #跑成績用
        # 检查文件是否是以.mp4为后缀
        print("file",file_name)
        if file_name.endswith(".txt"):
            video_num1.append(file_name)
            last_txt = video_num1[-1]
            print("file_name",file_name)
            txtname = "E://things/master/DIH/DIH/show2/"+video+"score.txt" #存成績的txt檔
            testfile = folder_path+"/test"
            print("len",len(video_num1))
            #要先改一下每個程式，讓它們可以直接回傳三個成績回來
            _name = last_txt.split(".")[0] #這次要處理的檔案名字 
                
            target_video = _name + ".mp4"
            print("target_video",target_video)
            if folder_path == "E://things/master/DIH/DIH/show2/01":
                print("11111")
                

                if (target_video in file_list) == True:
                    for_video_010.for_video01(_name)
                    print("here is video," + target_video)
                    sort_and_rename_files("E://things/master/DIH/DIH/show2/save/01",folder_path+"/"+target_video)
                    print(folder_path+"/"+file_name)
                    delete_txt(folder_path+"/"+file_name)
                    
                keypoint_file = folder_path+"/test/keypoints-3d-01"
                time_txt = folder_path+"/test/01.txt"
                print(len(video_num1))
                if len(video_num1)==6: 
                    list01 = []
                    stability, accuracy , smooth = for_score_01.score01(keypoint_file,time_txt)
                    list01.append(stability)
                    list01.append(accuracy)
                    list01.append(smooth)
                    print("lis",list01)
                    writetxt(list01,txtname)
                    
                    delete_txt(time_txt)
                    removefile(testfile) #把資料移除

                    video_num1 = []
    
                    print(folderpath)




            # elif folder_path == "E://things/master/DIH/DIH/show/04":
            #     list04 = []
            #     stability, accuracy,smooth = for_video_04.for_video04(video)
            #     list04.append(stability)
            #     list04.append(accuracy)
            #     list04.append(smooth)
            #     writetxt(list04,txtname)
            #     removefile(testfile) #把資料移除
            #     sort_and_rename_files("E://things/master/DIH/DIH/show/save/04",folder_path)
            #     delete_txt(folder_path)
            #     #delete_mp4_files(folder_path)
            #     #for_video_0402.   #呼叫04的程式   


            # elif folder_path == "E://things/master/DIH/DIH/show/08":
            #     list08 = []
            #     stability, accuracy,smooth = for_video_08.for_video08(video)
            #     list08.append(stability)
            #     list08.append(accuracy)
            #     list08.append(smooth)
            #     writetxt(list08,txtname)
            #     removefile(testfile) #把資料移除
            #     sort_and_rename_files("E://things/master/DIH/DIH/show/save/08",folder_path)
            #     delete_txt(folder_path)
            #     #delete_mp4_files(folder_path)
            #     #for_video_0802.   #呼叫08的程式 


            # elif folder_path == "E://things/master/DIH/DIH/show/09":
            #     list09 = []
            #     stability, accuracy,smooth = for_video_09.for_video09(video)
            #     list09.append(stability)
            #     list09.append(accuracy)
            #     list09.append(smooth)
            #     writetxt(list09,txtname)
            #     removefile(testfile) #把資料移除
            #     sort_and_rename_files("E://things/master/DIH/DIH/show/save/09",folder_path)
            #     delete_txt(folder_path)
            #     #delete_mp4_files(folder_path)
            #     #for_video_0902.   #呼叫09的程式 


            # elif folder_path == "E://things/master/DIH/DIH/show/12":
            #     list12 = []
            #     stability, accuracy,smooth = for_video_1202.for_video12(video)
            #     list12.append(stability)
            #     list12.append(accuracy)
            #     list12.append(smooth)
            #     writetxt(list12,txtname)
            #     removefile(testfile) #把資料移除
            #     sort_and_rename_files("E://things/master/DIH/DIH/show/save/12",folder_path)
            #     delete_txt(folder_path)
            #     #delete_mp4_files(folder_path)



            # elif folder_path == "E://things/master/DIH/DIH/show/14":
            #     list14 = []
            #     stability, accuracy,smooth = for_video_1402.for_video14(video)
            #     list14.append(stability)
            #     list14.append(accuracy)
            #     list14.append(smooth)
            #     writetxt(list14,txtname)
            #     removefile(testfile) #把資料移除
            #     sort_and_rename_files("E://things/master/DIH/DIH/show/save/14",folder_path)
            #     delete_txt(folder_path)
            #     #delete_mp4_files(folder_path)



    
    #return stability, accuracy, smooth


#if __name__ == '__main__':
while True:
    folder_list = ["01", "04", "08", "09", "12", "14"]

    for folder in folder_list:
        run_process(folder)
        print(folder)
    
    print("finish 1 time")

#run_process("01")
#run_process("04")
#run_process("08")
#run_process("09")
#run_process("12")
#run_process("14")
  # 等待1秒後再次檢查資料夾



print("good")