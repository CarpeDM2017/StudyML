# session01 Q1
num = input('Number of steps : ')
try: num = int(num)
except: print('Invalid input!')
for i in range(num):
	print( (num-i)*' '+"{0:<20}".format('*'*(i+1)))
