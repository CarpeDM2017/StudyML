def sol02(num):
    top = []
    mid = False
    asc = True
    num_stars = 2 - num % 2
    while (not mid):
        stars = '*' * num_stars
        space = ' ' * int((num - num_stars) / 2)
        top.append(space + stars + space)

        if num_stars == num:
            asc = False
        if asc:
            num_stars = min(num, num_stars + 4)
        else:
            if num_stars < len(top):
                mid = True
                break
            if num_stars == num:
                num_stars = num_stars - 4 + (num - 2 + num % 2) % 4
            else:
                num_stars -= 4
        if mid and num_stars > 2:
            length = len(top)
    for i in range(2, length + 1):
        top.append(top[length - i])

    for i in range(len(top)):
        print(top[i])
