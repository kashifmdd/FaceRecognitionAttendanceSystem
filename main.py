import os
import cv2
import face_recognition
import numpy as np
from datetime import datetime
import pandas as pd
import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk

class FaceRecognitionAttendanceSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition Attendance System")
        self.root.geometry("1200x650")
        
        # Initialize variables
        self.known_face_encodings = []
        self.known_face_names = []
        self.attendance_df = pd.DataFrame(columns=['Name', 'Date', 'Time'])
        self.camera_active = False
        self.video_capture = None
        
        # File paths
        self.faces_dir = "known_faces"
        self.attendance_file = "attendance_record.csv"
        
        # Create directories if they don't exist
        if not os.path.exists(self.faces_dir):
            os.makedirs(self.faces_dir)
            
        # Load existing attendance record if it exists
        if os.path.exists(self.attendance_file):
            self.attendance_df = pd.read_csv(self.attendance_file)
        
        # Load known faces
        self.load_known_faces()
        
        # Create UI
        self.create_ui()
    
    def load_known_faces(self):
        """Load known faces from the faces directory"""
        self.known_face_encodings = []
        self.known_face_names = []
        
        for filename in os.listdir(self.faces_dir):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                path = os.path.join(self.faces_dir, filename)
                name = os.path.splitext(filename)[0]
                
                # Load image and compute face encoding
                image = face_recognition.load_image_file(path)
                face_locations = face_recognition.face_locations(image)
                
                if face_locations:
                    face_encoding = face_recognition.face_encodings(image, face_locations)[0]
                    self.known_face_encodings.append(face_encoding)
                    self.known_face_names.append(name)
        
        print(f"Loaded {len(self.known_face_names)} known faces")
    
    def create_ui(self):
        """Create the user interface"""
        # Left frame for video feed
        left_frame = tk.Frame(self.root, width=600, height=600)
        left_frame.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.video_label = tk.Label(left_frame)
        self.video_label.pack(padx=10, pady=10)
        
        # Right frame for controls and info
        right_frame = tk.Frame(self.root, width=500, height=600)
        right_frame.pack(side=tk.RIGHT, padx=10, pady=10)
        
        # Title label
        title_label = tk.Label(right_frame, text="Face Recognition Attendance System", 
                              font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=10)
        
        # Status label
        self.status_label = tk.Label(right_frame, text="System Ready", font=("Arial", 12))
        self.status_label.grid(row=1, column=0, columnspan=2, pady=10)
        
        # Buttons
        start_btn = tk.Button(right_frame, text="Start Camera", command=self.start_camera, 
                             width=15, height=2, bg="#4CAF50", fg="white")
        start_btn.grid(row=2, column=0, pady=5, padx=5)
        
        stop_btn = tk.Button(right_frame, text="Stop Camera", command=self.stop_camera, 
                            width=15, height=2, bg="#F44336", fg="white")
        stop_btn.grid(row=2, column=1, pady=5, padx=5)
        
        register_btn = tk.Button(right_frame, text="Register New Face", command=self.register_new_face, 
                               width=15, height=2, bg="#2196F3", fg="white")
        register_btn.grid(row=3, column=0, pady=5, padx=5)
        
        view_btn = tk.Button(right_frame, text="View Attendance", command=self.view_attendance, 
                            width=15, height=2, bg="#FF9800", fg="white")
        view_btn.grid(row=3, column=1, pady=5, padx=5)
        
        export_btn = tk.Button(right_frame, text="Export Attendance", command=self.export_attendance, 
                              width=15, height=2, bg="#9C27B0", fg="white")
        export_btn.grid(row=4, column=0, pady=5, padx=5)
        
        exit_btn = tk.Button(right_frame, text="Exit", command=self.exit_application, 
                            width=15, height=2, bg="#607D8B", fg="white")
        exit_btn.grid(row=4, column=1, pady=5, padx=5)
        
        # Status text for recognized names
        self.recognized_label = tk.Label(right_frame, text="Recognized:", font=("Arial", 12))
        self.recognized_label.grid(row=5, column=0, columnspan=2, pady=(20, 5), sticky=tk.W)
        
        self.recognized_text = tk.Text(right_frame, height=10, width=40)
        self.recognized_text.grid(row=6, column=0, columnspan=2, pady=5)
    
    def start_camera(self):
        """Start the camera feed and face recognition"""
        if self.camera_active:
            return
        
        self.camera_active = True
        self.video_capture = cv2.VideoCapture(0)  # Use default camera
        self.status_label.config(text="Camera Active - Recognizing Faces")
        
        self.update_camera()
    
    def update_camera(self):
        """Update the camera feed and perform face recognition"""
        if not self.camera_active:
            return
        
        # Capture frame from camera
        ret, frame = self.video_capture.read()
        if not ret:
            self.status_label.config(text="Error accessing camera!")
            self.camera_active = False
            return
        
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Find faces in the frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        face_names = []
        recognized_names = set()
        
        for face_encoding in face_encodings:
            # Compare with known faces
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.6)
            name = "Unknown"
            
            # Use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
                    # Mark attendance
                    if name not in recognized_names:
                        self.mark_attendance(name)
                        recognized_names.add(name)
            
            face_names.append(name)
        
        # Display results on frame
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            # Draw box around face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Draw label with name
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        
        # Convert to format suitable for tkinter
        cv2_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2_image)
        imgtk = ImageTk.PhotoImage(image=img)
        
        # Update label
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
        
        # Update recognized names in the text box
        if recognized_names:
            now = datetime.now().strftime("%H:%M:%S")
            self.recognized_text.insert(tk.END, f"[{now}] Recognized: {', '.join(recognized_names)}\n")
            self.recognized_text.see(tk.END)
        
        # Schedule next update
        self.root.after(10, self.update_camera)
    
    def stop_camera(self):
        """Stop the camera feed"""
        if self.camera_active:
            self.camera_active = False
            if self.video_capture:
                self.video_capture.release()
            self.video_label.config(image='')
            self.status_label.config(text="Camera Stopped")
    
    def mark_attendance(self, name):
        """Mark attendance for a recognized person"""
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        
        # Check if already marked for today
        today_attendance = self.attendance_df[self.attendance_df['Date'] == date_str]
        if not any(today_attendance['Name'] == name):
            new_row = pd.DataFrame({'Name': [name], 'Date': [date_str], 'Time': [time_str]})
            self.attendance_df = pd.concat([self.attendance_df, new_row], ignore_index=True)
            self.attendance_df.to_csv(self.attendance_file, index=False)
            print(f"Marked attendance for {name}")
    
    def register_new_face(self):
        """Register a new face to the system"""
        # Stop camera if running
        was_active = self.camera_active
        if was_active:
            self.stop_camera()
        
        # Ask for name
        name_window = tk.Toplevel(self.root)
        name_window.title("Register New Face")
        name_window.geometry("300x150")
        
        tk.Label(name_window, text="Enter person's name:").pack(pady=10)
        name_entry = tk.Entry(name_window, width=25)
        name_entry.pack(pady=5)
        
        def take_photo():
            person_name = name_entry.get().strip()
            if not person_name:
                messagebox.showerror("Error", "Please enter a name")
                return
            
            # Take photo
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                messagebox.showerror("Error", "Could not access camera")
                return
            
            # Save image
            file_path = os.path.join(self.faces_dir, f"{person_name}.jpg")
            cv2.imwrite(file_path, frame)
            
            # Detect if face is present in the image
            image = face_recognition.load_image_file(file_path)
            face_locations = face_recognition.face_locations(image)
            
            if not face_locations:
                os.remove(file_path)
                messagebox.showerror("Error", "No face detected! Please try again.")
                return
            
            # Load the new face
            self.load_known_faces()
            
            messagebox.showinfo("Success", f"{person_name} has been registered successfully!")
            name_window.destroy()
            
            # Restart camera if it was active
            if was_active:
                self.start_camera()
        
        tk.Button(name_window, text="Take Photo", command=take_photo, 
                 bg="#2196F3", fg="white").pack(pady=10)
    
    def view_attendance(self):
        """View the attendance records"""
        if self.attendance_df.empty:
            messagebox.showinfo("Attendance", "No attendance records available")
            return
        
        # Create a window to display attendance
        attendance_window = tk.Toplevel(self.root)
        attendance_window.title("Attendance Records")
        attendance_window.geometry("600x400")
        
        # Create a frame for the table
        frame = tk.Frame(attendance_window)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create scrollbar
        scrollbar = tk.Scrollbar(frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create treeview
        columns = ("Name", "Date", "Time")
        tree = tk.ttk.Treeview(frame, columns=columns, show="headings", yscrollcommand=scrollbar.set)
        
        # Set column headings
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=100)
        
        # Add data to the table
        for _, row in self.attendance_df.iterrows():
            tree.insert("", tk.END, values=(row["Name"], row["Date"], row["Time"]))
        
        tree.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=tree.yview)
        
        # Add filter options
        filter_frame = tk.Frame(attendance_window)
        filter_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(filter_frame, text="Filter by date (YYYY-MM-DD):").pack(side=tk.LEFT, padx=5)
        date_entry = tk.Entry(filter_frame, width=15)
        date_entry.pack(side=tk.LEFT, padx=5)
        
        def apply_filter():
            date_filter = date_entry.get().strip()
            for item in tree.get_children():
                tree.delete(item)
            
            filtered_df = self.attendance_df
            if date_filter:
                filtered_df = filtered_df[filtered_df['Date'] == date_filter]
            
            for _, row in filtered_df.iterrows():
                tree.insert("", tk.END, values=(row["Name"], row["Date"], row["Time"]))
        
        tk.Button(filter_frame, text="Apply Filter", command=apply_filter).pack(side=tk.LEFT, padx=5)
        tk.Button(filter_frame, text="Clear Filter", command=lambda: (date_entry.delete(0, tk.END), apply_filter())).pack(side=tk.LEFT, padx=5)
    
    def export_attendance(self):
        """Export attendance records to CSV"""
        if self.attendance_df.empty:
            messagebox.showinfo("Export", "No attendance records to export")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Export Attendance"
        )
        
        if file_path:
            self.attendance_df.to_csv(file_path, index=False)
            messagebox.showinfo("Export", f"Attendance exported to {file_path}")
    
    def exit_application(self):
        """Exit the application"""
        if self.camera_active:
            self.stop_camera()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionAttendanceSystem(root)
    root.protocol("WM_DELETE_WINDOW", app.exit_application)
    root.mainloop()