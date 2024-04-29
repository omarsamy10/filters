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

# Function to draw a triangle using DDA algorithm
def draw_triangle_dda(x0, y0, x1, y1, x2, y2):
    lines = []
    lines.extend(draw_line_dda(x0, y0, x1, y1))
    lines.extend(draw_line_dda(x1, y1, x2, y2))
    lines.extend(draw_line_dda(x2, y2, x0, y0))
    return lines

# Example usage
x0, y0 = 1, 1
x1, y1 = 80, 40
x2, y2 = 40, 80

triangle_points = draw_triangle_dda(x0, y0, x1, y1, x2, y2)
print("Coordinates of the triangle using DDA algorithm:")
for point in triangle_points:
    print(point)

# Plotting the triangle
x_values, y_values = zip(*triangle_points)
plt.plot(x_values, y_values, marker='o')
plt.title('DDA Triangle Drawing Algorithm')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plt.show()