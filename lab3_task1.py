def AND(x1, x2):
    return x1 and x2
def OR(x1, x2):
    return x1 or x2
def NOT(x):
    return not x

def XOR(x1, x2):
    return OR(AND(x1, NOT(x2)), AND(NOT(x1), x2))

print(XOR(0, 0))
print(XOR(0, 1))
print(XOR(1, 0))
print(XOR(1, 1))
