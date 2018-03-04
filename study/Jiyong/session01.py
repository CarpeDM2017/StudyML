# Challenge Number 2

num = input('Number of steps : ')
try: num = int(num)
except: print('Invalid input!')

# Check whether the number is even
if num%2==0:
    star_start=2
    star_peak=1
else:
    star_start=1
    if num < 10:
        star_peak=1
    else:
        star_peak=2

# Initialize Parameters
star_list = [star_start]

while star_list[-1] < num:
    star_list.append(min(star_list[-1]+4, num))

print_list = star_list[:]
peak_count = 1
is_increasing = False
idx = -1

while not(peak_count == star_peak and print_list[-1] == print_list[0]):
    # Determine next index
    if is_increasing:
        idx += 1
    else:
        idx -= 1

    # Based on next index, Determine next direction
    # This does not change number to be appended
    if star_list[idx] == max(star_list):
        is_increasing = not is_increasing
        peak_count += 1
    elif star_list[idx] == star_list[1] and peak_count < star_peak:
        is_increasing = not is_increasing
    else :
        pass

    # Append selected number to list
    print_list.append(star_list[idx])

max_len = max(star_list)

for stars in print_list:
    space = (max_len-stars)/2
    print(" "*space+"*"*stars+" "*space)