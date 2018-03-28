def order(menu, number):
    assert isinstance(menu, str), "menu must be a string"
    assert isinstance(number, int), "number must be a integer"
    assert number > 0, "number must be positive"

    return "여기 {} {}인분이요!!".format(menu, number)

print(order("찡니를 보고싶은 굥이의 마음", 500))
print(order("베트남을 향한 굥이의 애정", 100))
print(order("클럽을 향한 설렘", 5050))
print(order("과제하기 싫은 나의 재능낭비", 9999))
