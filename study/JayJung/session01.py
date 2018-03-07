# session01.py

# easier assignment
num = input('Number of steps : ')
try: num = int(num)
except: print('Invalid input!')
for i in range(num):
	print( (num-i)*' '+"{0:<100}".format('*'*(i+1)))

# harder assignment
if num%2 == 1:
	for i in range(num+1):
		if (4*(i+1)-3) < num:
			j = (4*(i+1)-3)
			print( "{0:^30}".format('*'*j))
		else:
			k = copy(i)
			print( "{0:^30}".format('*'*num))
			if (4*(k-i)-3) > 5:
				j = (4*(i+1)-3)
				print( "{0:^30}".format('*'*j))
			else:
				print( "{0:^30}".format('*'*num))
else: