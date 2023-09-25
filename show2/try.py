import os
#import sys

if __name__ == '__main__':
    try:
        #file_path = "./try.txt"
        file_path = "E://things/master/DIH/DIH/show/try.txt"
        tet = "tryyyyy"
        with open(file_path, "w") as file:
            file.writelines(tet)
        print("finish")
        print("this is a test for unity")
    except:
        print("ERROR")

print("hete")