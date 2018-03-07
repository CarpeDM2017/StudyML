# session01.py

# easier assignment
num = input('Number of steps : ')
try: num = int(num)
except: print('Invalid input!')
for i in range(num):
	print( (num-i)*' '+"{0:<100}".format('*'*(i+1)))

# harder assignment
turn1 = False
turn2 = False
if num%2 == 1:
	for i in range(num+1):
		if not (turn2 == True):
			if not (turn1 == True):
				if (4*(i+1)-3) < num:
					j = (4*(i+1)-3)
					print( "{0:^30}".format('*'*j))
				else:
					print( "{0:^30}".format('*'*num))
					turn1 = True
					k1 = i
			else:
				if (4*(2*(k1)-i)+1) > 4:
					j = (4*(2*(k1)-i)+1)
					print( "{0:^30}".format('*'*j))
					l = (2*(k1)-i)+1
					k2 = i
				else:
					if (4*(l+(i-k2))-3) < num:
						j = (4*(l+(i-k2))-3)
						print( "{0:^30}".format('*'*j))
					else:
						print( "{0:^30}".format('*'*num))
						turn2 = True
						k = (4*(l+(i-k2))-3)
						k3 = i
		else:
			j = k + 4*(k3 - i)
			if (j) > 0:
				print( "{0:^30}".format('*'*j))
			else:
				pass



else:
	for i in range(num+1):
		if not (turn1 == True):
			if (4*(i+1)-2) < num:
				j = (4*(i+1)-2)
				print( "{0:^30}".format('*'*j))
			else:
				print( "{0:^30}".format('*'*num))
				turn1 = True
				k = i
		else:
			if (4*(2*(k+1)-i)-2) > 0:
				j = (4*(2*(k+1)-i)-6)
				print( "{0:^30}".format('*'*j))
			else:
				pass
