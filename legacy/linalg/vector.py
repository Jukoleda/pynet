
def dot(vector, row):
    value = 0
    for x,w in zip(vector, row):
        value += x * w
    return value
    