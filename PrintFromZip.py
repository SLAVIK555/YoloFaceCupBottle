import zipfile

word = "test"

zip_archive = zipfile.ZipFile(word + ".zip", "r")

filelist = []

for file_info in zip_archive.infolist(): 
	if ".jpg" in file_info.filename:
		print(file_info.filename)
		filelist.append("data/" + word + "/" + file_info.filename)

textfile = open(word + ".txt", 'w')

for file in filelist:
	textfile.write(file + "\n")

textfile.close()