wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username=ikami&password=duong1998&submit=Login' https://www.cityscapes-dataset.com/login/
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=1
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=3
unzip gtFine_trainvaltest.zip
unzip leftImg8bit_trainvaltest.zip
