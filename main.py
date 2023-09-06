import numpy as np
import csv
import matplotlib.pyplot as plt

# Load the sensor data from your provided text
data = []
with open('slam_task.csv', 'r') as file:
    for line in file:
        line = line.strip()  # Remove leading/trailing whitespace
        if not line or line.startswith("#"):
            continue  # Skip empty lines and comments
        time, vel, pos = map(float, line.split(','))
        data.append((time, vel, pos))

# Initialize Kalman filter variables
A = np.array([[1, 1], [0, 1]])  # State transition matrix
H = np.array([[1, 0], [0, 1]])  # Measurement matrix
x_hat = np.array([[0], [0]])  # Initial state estimate
P = np.array([[10, 0], [0, 10]])  # Initial error covariance
Q = np.array([[59, 0], [0, 8.5]])  # Process noise covariance
R = np.array([[1, 0], [0, 0.5]])  # Measurement noise covariance

# Lists to store the estimated results
estimated_velocities = []
estimated_positions = []

# Open a CSV file for writing the results
with open('kalman_results.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)

    # Write the CSV header
    csv_writer.writerow(['Time', 'Estimated Vel', 'Estimated Pos'])

    for time, vel, pos in data:
        # Prediction step
        x_hat = np.dot(A, x_hat)
        P = np.dot(np.dot(A, P), A.T) + Q

        # Update step
        y = np.array([[vel], [pos]]) - np.dot(H, x_hat)
        S = np.dot(np.dot(H, P), H.T) + R
        K = np.dot(np.dot(P, H.T), np.linalg.inv(S))
        x_hat = x_hat + np.dot(K, y)
        P = np.dot((np.eye(2) - np.dot(K, H)), P)

        # Append the results to the CSV file
        csv_writer.writerow([time, x_hat[0][0], x_hat[1][0]])

        # Append the estimated velocity and position to the lists
        estimated_velocities.append(x_hat[0][0])
        estimated_positions.append(x_hat[1][0])

# Plot the estimated velocity
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot([time for time, _, _ in data], [vel for _, vel, _ in data], label='Measured Velocity', marker='o', linestyle='-', color='b')
plt.plot([time for time, _, _ in data], estimated_velocities, label='Estimated Velocity', marker='x', linestyle='--', color='r')
plt.xlabel('Time')
plt.ylabel('Velocity')
plt.legend()

# Plot the estimated position
plt.subplot(2, 1, 2)
plt.plot([time for time, _, _ in data], [pos for _, _, pos in data], label='Measured Position', marker='o', linestyle='-', color='b')
plt.plot([time for time, _, _ in data], estimated_positions, label='Estimated Position', marker='x', linestyle='--', color='r')
plt.xlabel('Time')
plt.ylabel('Position')
plt.legend()

plt.tight_layout()
plt.show()