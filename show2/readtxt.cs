//這個檔案是拿來讀存有成績的txt檔給unity的
using UnityEngine;
using System.IO;

public class ReadAndClearTxtFile : MonoBehaviour
{
    public string filePath= "./allscore.txt"; // 檔案路徑

    void Start()
    {
        //檢查檔案是否存在
        if (File.Exists(filePath))
        {
            // 讀取檔案內容
            string fileContent = File.ReadAllText(filePath);

            // 處理檔案內容，例如顯示在 Console 或進行其他處理
            Debug.Log("File content: " + fileContent);

            // 清除檔案內容
            File.WriteAllText(filePath, string.Empty);
        }
        else
        {
            Debug.LogWarning("File not found: " + filePath);
        }
    }
}
