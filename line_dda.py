import matplotlib.pyplot as plt

# Function to draw a line using DDA algorithm
def draw_line_dda(x0, y0, x1, y1):
    dx = x1 - x0
    dy = y1 - y0
    steps = abs(dx) if abs(dx) > abs(dy) else abs(dy)
    x_increment = dx / steps
    y_increment = dy / steps
    x, y = x0, y0
    points = [(round(x), round(y))]

    for _ in range(steps):
        x += x_increment
        y += y_increment
        points.append((round(x), round(y)))

    return points

# Example usage
x0, y0 = 1, 1
x1, y1 = 50, 40

points = draw_line_dda(x0, y0, x1, y1)
print("Coordinates of the line using DDA algorithm:")
for point in points:
    print(point)

# Plotting the line
x_values, y_values = zip(*points)
plt.plot(x_values, y_values, marker='o')
plt.title('DDA Line Drawing Algorithm')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plt.show()

