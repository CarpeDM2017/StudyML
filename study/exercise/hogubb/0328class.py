new_list = [True, 1, 'True']
new_tuple = (False, 0, 'False')
new_dict = {1 : False, 0 : True}

print(new_list[0] + new_tuple[1])   # True + 0 == 1
print(new_dict[0] + new_tuple[1])   # True + 0 == 1
print(new_list[2] + new_tuple[2])   # 'True' + 'False' == 'TrueFalse'
print(new_list[1] + new_dict[0])

list_2d = list()
for i in range(1,5):
    list_2d.append(list(range(i)))
print(list_2d)

my_list=[1,2,3]
my_list[3]=4 #안됨

new_list = [True, 0, abs(-1), 2, 3]
new_list = sorted(new_list) # 오름차순 정렬
print(new_list)
all(new_list)   # 얘네가 모두 true니??
any(new_list)   # list안의 값 중에 true 있니?
min(new_list)   #  제일 작은 값
max(new_list)   #  제일 큰 값
len(new_list)   # length
for i, j in enumerate(new_list):    # 1번2번3번 이렇게 세는 거~!0 0
my_list=[1,2,5,5,3]
my_set=set([1,2,5,5,3])

for i,j in enumerate(my_list):
    print(i,j) #i에 01234, j에 12553

for i,j in enumerate(my_set):
    print(i,j) #i에 0123, j에 1235 << 오름차순 정렬


new_list[1] == new_list[2]
id(new_list[1]) == id(new_list[2])

int('3') #숫자 3
str(3) #문자3
int('101',2) #101이 이진법으로 5라서 5나옴
int('F34',16) #F34는 16진법 수인데 10진법으로 바꾸면 3892

list1 = list(range(3))
print(list1)
list2 = list(range(3,6))
print(list2)
list(zip(list1, list2))

#def=정의한다, 함수를 직접 만듦

def order(menu, number):
    # 입력값에 대한 예외처리용 구문
    assert isinstance(menu, str), "menu must be a string"
    assert isinstance(number, int), "number must be an integer"
    assert number > 0, "number must be positive"

    return "여기 {} {}인분이요!!".format(menu, number)

print(order("돼지갈비", 3))    # 여기 돼지갈비 3인분이요!!
print(order("파이썬", 20))    # 여기 파이썬 20인분이요!!
print(order(3, "돼지갈비"))     # AssertionError
