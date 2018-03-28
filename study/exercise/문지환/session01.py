def easy():
    num = input("Number of steps: ")
    try: num = int(num)
    except: print("Invalid Input!")

    for i in range(num):
        print(' '* (num - i - 1) + '*' * (i + 1))


def difficult():
    num = input("Number of steps: ")
    try: num = int(num)
    except: print("Invalid Input!")

    stars = [2] if num % 2 == 0 else [1]
    while(stars[-1] < num):
        if num - stars[-1] >= 4:
            stars.append(stars[-1] + 4)
        else:
            stars.append(stars[-1] + 2)

    stars += reversed(stars[:-1])

    if num > 10 and num % 2 == 1:
        stars.pop()
        stars += reversed(stars[:-1])

    for star in stars:
        blanks = num - star
        print(' '*(blanks//2) + '*'*star + ' '*(blanks//2))


if __name__ == '__main__':
    import sys
    try:
        if sys.argv[1] == 'easy':
            easy()
        elif sys.argv[1] == 'difficult':
            difficult()
    
    except:
        easy()
