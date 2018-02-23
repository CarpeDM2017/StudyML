## Session 02
#### 함수와 클래스

* Python 기초문법 연습
* Python의 내장함수
* 함수와 클래스
* Numpy 패키지
* Matplotlib 패키지


1. (제출 코드 설명)  


#### Python 기초문법 연습

| 자료형                  | 설명                           | 연산                     |
| ----------------------- | ------------------------------ | ------------------------ |
| <b>논리(bool)</b>       | 참(1) 또는 거짓(0)             | or(+), and(\*)           |
| <b>숫자(int, float)</b> | 정수 또는 실수                 | +, -, \*, /, %           |
| <b>문자열(str)</b>      | 단어, 문자들의 집합            | +, \*, [ ]               |
| <b>리스트(list)</b>     | [list1, list2, list3, list4]   | +, \*, [ ], del, append  |
| <b>튜플(tuple)</b>      | (tuple1, tuple2, tuple3)       | +, \*, [ ]               |
| <b>딕셔너리(dict)</b>   | {key1 : value1, key2 : value2} | [ ], keys(), values()    |
| <b>집합(set)</b>        | {set1, set2, set3, set4}       | intersection, union, add |
| <b>없음(None)</b>       | 그 어떤 자료형도 아닌 경우     |                          |

지난 스터디에 이어 배열 구조의 자료형을 다루어봅시다.
```Python
new_list = [True, 1, 'True']
new_tuple = (False, 0, 'False')
new_dict = {1 : False, 0 : True}

print(new_list[0] + new_tuple[1])   # True + 0 == 1
print(new_dict[0] + new_tuple[1])   # True + 0 == 1
print(new_list[2] + new_tuple[2])   # 'True' + 'False' == 'TrueFalse'
print(new_list[1] + new_dict[0])    # 1 + True == 2

new_list[0] = None  # new_list = [None, 1, 'True']
new_tuple[1] = None # TypeError
new_dict[1] = None  # new_dict = {0 : True, 1 : None}

new_set = set('FnC' + 'CarpeDM')    # {'C', 'D', 'F', 'M', 'a', 'e', 'n', 'p', 'r'}
```
배열은 다른 배열을 그 원소로 가질 수 있습니다.
```Python
list_2d = list()
for i in range(1,5):
    list_2d.append(list(range(i)))
print(list_2d)  # [[0], [0, 1], [0, 1, 2], [0, 1, 2, 3]]

list_2d[2][1] = -1
print(list_2d)  # [[0], [0, 1], [0, -1, 2], [0, 1, 2, 3]]
```

#### Python의 내장함수

Python에서는 print, input, range, append와 같이 자료형을 보다 편리하게 다루기 위한 여러가지 함수를 자체적으로 제공합니다. 여기서 <b>함수</b>란 어떤 자료형을 입력값으로 주었을 때, 정해진 결과값을 돌려주는 관계를 의미합니다. 함수를 정의하는 문법을 살펴보기 전에, 자주 사용하는 내장함수들에는 어떤 것이 있는지 한번 살펴봅시다.
```Python
new_list = [True, 0, abs(-1), 2, 3]
new_list = sorted(new_list)     # new_list = [0, True, 1, 2, 3]
all(new_list)   # False
any(new_list)   # True
min(new_list)   # 0
max(new_list)   # 3
len(new_list)   # 5

for i, j in enumerate(new_list):    # 0 0
    print(i,j)                      # 1 True
                                    # 2 1
                                    # 3 2
                                    # 4 3

new_list[1] == new_list[2]          # True
id(new_list[1]) == id(new_list[2])  # False

3 == int('3')   # True
str(3) == '3'   # True
int('101', 2)   # 5
int('F34', 16)  # 3892

type(3) # int
isinstance(3, float)    # False
3 == float(3)   # True

list1 = list(range(3))
list2 = list(range(3,6))
list(zip(list1, list2)) # [(0, 3), (1, 4), (2, 5)]
```

#### 함수와 클래스

* ##### 함수
Python에서 내장함수로 제공하지 않는 특정한 기능을 여러번 반복해서 수행하고 싶을 때, 직접 함수를 정의해 이를 활용할 수 있습니다. Python에서의 함수는 def 명령어를 통해 정의합니다. 입력값이 될 변수를 괄호 안에 적고, 출력값을 def 구문 안에 있는 return 명령어 뒤에 적습니다.

```Python
def order(menu, number):
    # 입력값에 대한 예외처리용 구문
    assert isinstance(menu, str), "menu must be a string"
    assert isinstance(number, int), "number must be an integer"
    assert number > 0, "number must be positive"

    return "여기 {} {}인분이요!!".format(menu, number)

print(order("돼지갈비", 3))    # 여기 돼지갈비 3인분이요!!
print(order("파이썬", 20))    # 여기 파이썬 20인분이요!!
print(order(3, "돼지갈비"))     # AssertionError

type(order("파이썬", 20))  # str
```
이때 함수는 입력값이나 출력값을 가지지 않을 수 있습니다. 이때 출력값의 자료형은 None형입니다.
```Python
def order():
    print("주문하시겠어요?")

order()             # 주문하시겠어요?
print(order())      # 주문하시겠어요? / None
type(order())       # 주문하시겠어요? / NoneType
```

* ##### 클래스
![session02_01](./image/session02_01_cookie.jpg)
<br></br>
쿠키틀을 한번 만들어두면 같은 모양의 쿠키를 계속 만들어낼 수 있듯이, 동일한 형태의 변수와 함수를 계속해서 생성하기 위해 Python에서는 <b>클래스</b>라는 도구를 이용합니다. 클래스라는 쿠키틀로 찍어낸 쿠키를 <b>객체</b> 또는 <b>인스턴스</b>라고 부릅니다. 같은 클래스의 객체는 같은 구조를 가지고 있지만, 저장되어 있는 변수값은 서로 다를 수 있습니다.

```Python
class Chicken:
    def __init__(self, menu):

        self.menus = {'후라이드' : 15000, '양념' : 20000, '반반' : 18000}

        assert isinstance(menu, str), "menu must be a string"
        assert menu in self.menus.keys(), "해당 메뉴는 현재 서비스하고 있지 않습니다"

        self.menu = menu

    def order(self, number):
        assert isinstance(number, int), "number must be an integer"
        assert number > 0, "number must be positive"

        self.number = number
        print("여기 {} {}인분이요!!".format(self.menu, number))

    def calculate(self):
        total = self.menus[self.menu] * self.number
        print("{}원 되시겠습니다.".format(total))

chicken1 = Chicken("양념")
chicken1.order(3)                   # 여기 양념 3인분이요!!
chicken1.calculate()                # 60000원 되시겠습니다.
chicken1.menus["파닭"] = 25000
chicken1.menu = "파닭"
chicken1.calculate()                # 75000원 되시겠습니다.

chicken2 = Chicken("파닭")           # AssertionError
```
여기서 chicken1, chicken2는 Chicken 클래스의 객체들이고, 클래스의 정의에 따라 \__init\__, order, calculate 함수와 menus, menu, number 변수를 가지고 있습니다. 이때 order와 같이 클래스의 객체가 가지고 있는 함수를 <b>메소드</b>라고 부릅니다. \__init\__ 메소드와 같이 \__로 시작하는 메소드는 그 자체로 특수한 기능을 가지고 있는데요, 주로 객체의 생성, 복사, 출력 등과 관련한 작업을 자동으로 수행해줍니다.

#### Numpy 패키지

#### Matplotlib 패키지

#### 참고자료

* 객체 지향 프로그래밍 - 위키백과, 우리 모두의 백과사전
https://ko.wikipedia.org/wiki/%EA%B0%9D%EC%B2%B4_%EC%A7%80%ED%96%A5_%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%98%EB%B0%8D
* Quickstart tutorial — NumPy v1.13.dev0 Manual  
https://docs.scipy.org/doc/numpy-dev/user/quickstart.html
* Pyplot tutorial — Matplotlib 2.0.2 documentation  
https://matplotlib.org/users/pyplot_tutorial.html
* 점프 투 파이썬 - WikiDocs  
https://wikidocs.net/book/1
* Learn to code | Codecademy  
https://www.codecademy.com/
