a = input ("what is a?")
if (a == '0'):
    print ("a is zero")
else:
    print ("a is not zero")

name = "seonwoo"
character = ["cute","pretty","smart"]

for i in character :
    print (name, 'is', i)

for i in range(1,4):
    for j in range(1,4):
        print (10*i + j)
