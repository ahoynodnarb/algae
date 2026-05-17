import symdiff as sd

x = sd.Variable("x")
y = sd.Variable("y")
z = sd.pow(sd.tan(sd.exp(x - 2) * (sd.log(y))), sd.csc((y - 7) * 3) + 2 / sd.atan(x))
print(z)
x.eval(1)
y.eval(2)
print(z.eval())
