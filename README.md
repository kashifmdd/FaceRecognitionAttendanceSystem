# 📸 Face Recognition Attendance System

Ek real-time attendance system jo webcam ka istemaal karke chehre pehchaanta hai aur attendance ko ek CSV file mein log karta hai.

---

## 📜 Description
Yeh project Python ka istemaal karke ek automatic attendance system banata hai. Yeh `opencv` ka istemaal webcam feed lene ke liye aur `face_recognition` library ka istemaal chehron ko pehchaanne ke liye karta hai. Jab koi jaana-pehchaana chehra (known face) detect hota hai, toh yeh system uska naam aur time ek `attendance.csv` file mein save kar deta hai.

---

## ✨ Features (Mukhya Visheshtayein)
* **Real-time Detection:** Webcam se live video mein chehron ko detect karta hai.
* **Automatic Logging:** Pehchaane gaye chehron ka naam aur time `attendance.csv` file mein automatically save karta hai.
* **Duplicate Entry Prevention:** Din mein ek student ki attendance ek baar hi lagata hai. (Agar aapne yeh feature banaya hai toh)
* **Easy to Add Users:** Naye student ko add karne ke liye bas unki ek photo `known_images` folder mein daalni hai.

---

## 💻 Tech Stack (Istemal Ki Gayi Technology)
* **Python 3.10**
* **OpenCV-Python** (Webcam feed aur image processing ke liye)
* **face_recognition** (dlib) (Core face recognition logic ke liye)
* **Pandas** (CSV file ko padhne aur likhne ke liye)
* **Conda** (Environment management ke liye)

---

## 🚀 Installation & Setup

Is project ko apne local system par chalaane ke liye yeh steps follow karein:

**1. Repository ko Clone Karein:**
```bash
git clone https://github.com/kashifmdd/FaceRecognitionAttendanceSystem.git
cd FaceRecognitionAttendanceSystem