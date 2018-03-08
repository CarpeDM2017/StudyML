# ASSIGNMENT
num = input ('Number of steps : ')
try : num = int(num)
except : print ('Invalid input!')

for i in range(num) :
    print(""*(num-(i+1))+"*"*(i+1))
