num= input('Number of steps : ')
try: num = int(num)
except: print('Invalid input!')
#여기서부터 작성해주시면 됩니다.

for i in range(1, num+1):
    spy=num+1-i
    for j in range(spy-1):
        print(" ", end="")
    for k in range(i):
        print("*", end="")
              
    print()