"""
2018.04.04
My answers for problems in session01

"""
num = input('Number of steps : ')
try: num = int(num)
except: print('Invalid input!')


def Q1(num):
    for i in range(num):
        space = ' '*(num-i-1)
        stars = '*'*(i+1)
        print(space+stars)


def Q2(num):
    top = []
    isHalf = False
    isAscend = True
    num_stars = 2-num%2
    while(not isHalf):
        stars = '*'*num_stars
        space = ' '*int((num-num_stars)/2)
        top.append(space+stars+space)

        if num_stars == num:
            isAscend = False
        if isAscend:
            num_stars = min(num, num_stars+4)
        else:
            if num_stars < len(top):
                isHalf = True
                break
            if num_stars == num:
                num_stars = num_stars - 4 + (num-2+num%2)%4
            else:
                num_stars -= 4

    if isHalf and num_stars > 2:
        length = len(top)
        for i in range(2,length+1):
            top.append(top[length-i])

    for i in range(len(top)):
        print(top[i])
