<<<<<<< HEAD
num = input('Number of steps :')
try : num = int(num)
except : print('Invalid input!')
for i in range(num):
    print(' '*(num+1-i)+'*'*(i+1))
=======

num = input('Number of steps : ')
try: num = int(num)
except: print('Invalid input!')
for i in range(num) :
    print(" "*(num-1-i) + '*'*(i+1))
>>>>>>> 9140962d0af5bed17f64368eba1f885a992c6269
