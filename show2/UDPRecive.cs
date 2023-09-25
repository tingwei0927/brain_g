using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.Text;
using System.Net;
using System.Net.Sockets;
using System.Threading;

public class UDPRecive : MonoBehaviour
{
    Thread receiveThread;
    UdpClient client; 
    public int port = 5714;
    public bool startRecieving = true;
    public string data;
    // Start is called before the first frame update
    void Start()
    {
        receiveThread = new Thread(
            new ThreadStart(ReceiveData));
        receiveThread.IsBackground = true;
        receiveThread.Start();
        
    }

    // Update is called once per frame
    void Update()
    {
        
    }
    
    private void ReceiveData()
    {
        client = new UdpClient(port);
        while (startRecieving)
        {
             Debug.Log("startRecieving");
            try
            {
                IPEndPoint anyIP = new IPEndPoint(IPAddress.Any, 0);
                byte[] dataByte = client.Receive(ref anyIP);
                data = Encoding.UTF8.GetString(dataByte);
                Debug.Log(data);
            }
            catch (Exception err)
            {
                print(err.ToString());
            }
        }
    }
}