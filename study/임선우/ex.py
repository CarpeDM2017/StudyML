num = input('Number of steps : ')
try: num = int(num)
except: print('Invalid input!')
print ("Number of steps : "+ str(num) )
for i in range(num):
    a = " "*(num-i+1)+"*"*(i+1)
    print(a)
