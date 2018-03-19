num = input('number of steps : ')
try : num = int(num)
except : print('invalid input!')
for i in range(num):
    print(' '*(num+1-i)+'*'*(i+1))
