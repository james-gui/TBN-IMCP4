import numpy as np

research = 0
scale = 0
speed = 0
maxPnL = 0

for x in range(1, 101):
    for y in range(0, 101-x):
        z = 100-x-y
        pnl = (200_000 * np.log(1+x) / np.log(101)) * (0.07 * y) * (0.008 * z + 0.1) - 50_000
        if pnl > maxPnL:
            research = x
            scale = y
            speed = z
            maxPnL = pnl

print(f"Research: {research}, Scale: {scale}, Speed: {speed}")
print(f"Total PnL: {maxPnL}")
