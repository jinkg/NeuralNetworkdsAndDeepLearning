B = 1000000000
M = 1000000

data = B
for _ in range(M):
    data += 0.000001

print(data - B)
