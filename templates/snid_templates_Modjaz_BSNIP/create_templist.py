import os

files = os.listdir('.')

f = open('templist.txt', 'w')
for i in files:
    f.write(i)
    f.write('\n')

f.close()
    
