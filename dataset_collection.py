import serial
import time
import matplotlib.pyplot as plt
import csv
import os
import datetime
import pandas as pd
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


class SmartGloveApp:
    def __init__(self):
        self.csv_text = "time(s),thumb_finger,index_finger,middle_finger,ring_finger,little_finger,gesture\n"
        self.start_log = False
        self.command_bytes = bytearray(16)
        self.command_is_ready = False
        self.com_counter = 0

        self.thumb_finger_voltage = 0
        self.index_finger_voltage = 0
        self.middle_finger_voltage = 0
        self.ring_finger_voltage = 0
        self.little_finger_voltage = 0
        self.time = 0

        self.serial_port = serial.Serial()
        self.serial_port.baudrate = 9600
        self.serial_port.timeout = 1

        self.init_virtual_com_port()

        # Initialize plot
        self.fig, self.ax = plt.subplots()
        self.thumb_line, = self.ax.plot([], [], label='Thumb Finger Voltage')
        self.index_line, = self.ax.plot([], [], label='Index Finger Voltage')
        self.middle_line, = self.ax.plot([], [], label='Middle Finger Voltage')
        self.ring_line, = self.ax.plot([], [], label='Ring Finger Voltage')
        self.little_line, = self.ax.plot([], [], label='Little Finger Voltage')
        self.ax.legend()
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 3.3)

        self.thumb_data = []
        self.index_data = []
        self.middle_data = []
        self.ring_data = []
        self.little_data = []
        self.time_data = []

        self.gesture_templates = {}  # Dictionary to hold gesture templates
        self.current_gesture_data = []  # List to store currently collected gesture data

    def init_virtual_com_port(self):
        self.serial_port.port = 'COM5'  # Replace 'COM5' with your actual COM port
        self.serial_port.open()

    def calculate_checksum(self, data_bytes):
        return sum(data_bytes[:-1]) & 0xFF

    def send_no_operation_command(self):
        self.command_bytes[0] = 0x01
        self.command_bytes[1] = self.com_counter % 256
        self.command_bytes[2:15] = [0x00] * 13
        self.command_bytes[15] = self.calculate_checksum(self.command_bytes)
        self.serial_port.write(self.command_bytes)

    def read_data(self):
        if self.serial_port.in_waiting >= 16:
            rx_bytes = self.serial_port.read(16)
            self.time += 0.001
            self.timestamp = datetime.datetime.now()

            self.show_received_bytes(rx_bytes)
            self.update_plot()

            # Collect real-time gesture data
            self.current_gesture_data.append({
                "time(s)": self.time,
                "thumb_finger": self.thumb_finger_voltage,
                "index_finger": self.index_finger_voltage,
                "middle_finger": self.middle_finger_voltage,
                "ring_finger": self.ring_finger_voltage,
                "little_finger": self.little_finger_voltage
            })

            if self.start_log:
                recognized_gesture, min_distance = self.recognize_gesture(self.current_gesture_data)
                print(f"Recognized Gesture: {recognized_gesture} with DTW distance: {min_distance}")
                self.csv_text += f"{self.time},{self.thumb_finger_voltage},{self.index_finger_voltage},{self.middle_finger_voltage},{self.ring_finger_voltage},{self.little_finger_voltage},{recognized_gesture}\n"

    def show_received_bytes(self, rx_bytes):
        thumb_finger = (rx_bytes[2] << 8) + rx_bytes[1]
        self.thumb_finger_voltage = (thumb_finger * 3.3) / 4096

        index_finger = (rx_bytes[4] << 8) + rx_bytes[3]
        self.index_finger_voltage = (index_finger * 3.3) / 4096

        middle_finger = (rx_bytes[6] << 8) + rx_bytes[5]
        self.middle_finger_voltage = (middle_finger * 3.3) / 4096

        ring_finger = (rx_bytes[10] << 8) + rx_bytes[9]
        self.ring_finger_voltage = (ring_finger * 3.3) / 4096

        little_finger = (rx_bytes[12] << 8) + rx_bytes[11]
        self.little_finger_voltage = (little_finger * 3.3) / 4096

        print(f"thumb_finger_voltage: {self.thumb_finger_voltage:.2f} V")
        print(f"index_finger_voltage: {self.index_finger_voltage:.2f} V")
        print(f"middle_finger_voltage: {self.middle_finger_voltage:.2f} V")
        print(f"ring_finger_voltage: {self.ring_finger_voltage:.2f} V")
        print(f"little_finger_voltage: {self.little_finger_voltage:.2f} V")

    def update_plot(self):
        self.time_data.append(self.time)
        self.thumb_data.append(self.thumb_finger_voltage)
        self.index_data.append(self.index_finger_voltage)
        self.middle_data.append(self.middle_finger_voltage)
        self.ring_data.append(self.ring_finger_voltage)
        self.little_data.append(self.little_finger_voltage)

        self.thumb_line.set_data(self.time_data, self.thumb_data)
        self.index_line.set_data(self.time_data, self.index_data)
        self.middle_line.set_data(self.time_data, self.middle_data)
        self.ring_line.set_data(self.time_data, self.ring_data)
        self.little_line.set_data(self.time_data, self.little_data)

        if self.time > self.ax.get_xlim()[1]:
            self.ax.set_xlim(self.ax.get_xlim()[0] + 5, self.ax.get_xlim()[1] + 5)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def recognize_gesture(self, current_gesture_data):
        # Convert the current gesture data into format suitable for DTW comparison
        current_gesture = {
            "thumb_finger": [x["thumb_finger"] for x in current_gesture_data],
            "index_finger": [x["index_finger"] for x in current_gesture_data],
            "middle_finger": [x["middle_finger"] for x in current_gesture_data],
            "ring_finger": [x["ring_finger"] for x in current_gesture_data],
            "little_finger": [x["little_finger"] for x in current_gesture_data]
        }

        min_distance = float('inf')
        recognized_gesture = "unknown"

        if not self.gesture_templates:
            return recognized_gesture, min_distance

        for gesture_name, template in self.gesture_templates.items():
            total_distance = 0
            for finger in template.keys():
                distance, _ = fastdtw(current_gesture[finger], template[finger], dist=euclidean)
                total_distance += distance

            if total_distance < min_distance:
                min_distance = total_distance
                recognized_gesture = gesture_name

        return recognized_gesture, min_distance

    def add_gesture_template(self, gesture_name, example_data):
        template = {
            "thumb_finger": example_data["thumb_finger"],
            "index_finger": example_data["index_finger"],
            "middle_finger": example_data["middle_finger"],
            "ring_finger": example_data["ring_finger"],
            "little_finger": example_data["little_finger"]
        }
        self.gesture_templates[gesture_name] = template

    def collect_gesture_data(self, gesture_name):
        """
        Collects data for a specific gesture and adds it to the gesture templates.
        """
        self.current_gesture_data = []  # Clear any previous data

        print(f"Starting to collect data for gesture: {gesture_name}")
        start_time = time.time()

        # Collect data until a stopping condition is met
        while True:
            self.read_data()
            # Append the current gesture data
            current_gesture_data_point = {
                "time(s)": self.time,
                "thumb_finger": self.thumb_finger_voltage,
                "index_finger": self.index_finger_voltage,
                "middle_finger": self.middle_finger_voltage,
                "ring_finger": self.ring_finger_voltage,
                "little_finger": self.little_finger_voltage
            }
            self.current_gesture_data.append(current_gesture_data_point)

            # Optionally, you can add a condition to stop collecting data
            # For example, after a specific number of data points or based on user input

            time.sleep(0.01)  # Adjust sleep time as needed

            # For example, stop collecting data after 5 seconds
            if time.time() - start_time > 5:
                break

        # Prepare data to be used as a gesture template
        example_data = {
            "thumb_finger": [x["thumb_finger"] for x in self.current_gesture_data],
            "index_finger": [x["index_finger"] for x in self.current_gesture_data],
            "middle_finger": [x["middle_finger"] for x in self.current_gesture_data],
            "ring_finger": [x["ring_finger"] for x in self.current_gesture_data],
            "little_finger": [x["little_finger"] for x in self.current_gesture_data]
        }
        self.add_gesture_template(gesture_name, example_data)
        print(f"Gesture '{gesture_name}' template added.")

    def start_logging(self):
        self.csv_text = "time(s),thumb_finger,index_finger,middle_finger,ring_finger,little_finger,gesture\n"
        self.start_log = True

    def stop_logging(self):
        self.start_log = False
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"log_{current_time}.csv"
        with open(os.path.join(os.getcwd(), file_name), 'w') as file:
            file.write(self.csv_text)
        print(f"Data saved to {file_name}")

    def main_loop(self):
        self.start_logging()  # Start logging at the beginning
        plt.ion()
        try:
            while True:
                self.com_counter += 1
                if not self.command_is_ready:
                    self.send_no_operation_command()
                else:
                    self.command_is_ready = False

                self.read_data()
                time.sleep(0.01)
        except KeyboardInterrupt:
            self.stop_logging()
            print("Logging stopped and data saved.")

    def plot_csv(file_name):
        # Read the CSV file
        data = pd.read_csv(file_name)

        # Plot each column
        plt.figure(figsize=(10, 6))
        plt.plot(data['time(s)'], data['thumb_finger'], label='Thumb Finger Voltage')
        plt.plot(data['time(s)'], data['index_finger'], label='Index Finger Voltage')
        plt.plot(data['time(s)'], data['middle_finger'], label='Middle Finger Voltage')
        plt.plot(data['time(s)'], data['ring_finger'], label='Ring Finger Voltage')
        plt.plot(data['time(s)'], data['little_finger'], label='Little Finger Voltage')

        # Customize the plot
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (V)')
        plt.title('Finger Voltages Over Time')
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    app = SmartGloveApp()

    # Example of collecting and labeling a gesture
    print("Collecting data for gesture 'hand_open'")
    app.collect_gesture_data("hand_open")

    # Start the main loop for real-time data collection and recognition
    app.main_loop()
