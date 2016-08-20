filenames = []
tempfilelist = 'templist.txt'

with open(tempfilelist) as FileObj:
    for lines in FileObj:
        filename = lines.strip('C:\Users\Daniel\OneDrive\Documents\Thesis Project\scripts\SNClassifying\templates\superfit_templates\sne\\').strip('\n')
        if (filename.split('.')[1][0] == 'u'):
            continue
        filenames.append(filename)
        
        

f = open(tempfilelist, 'w')
for i in filenames:
    f.write(i)
    f.write('\n')

f.close()
