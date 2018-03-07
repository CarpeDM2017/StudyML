num = input('Number of steps :')
try : num = int(num)
except : print('Invalid input!')
for i in range(num):
    print(' '*(num+1-i)+'*'*(i+1))
