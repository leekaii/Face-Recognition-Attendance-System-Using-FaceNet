# Face Recognition Attendance System Using FaceNet
Build this interactive project in a team of three other talented individuals: Christopher, Rejoy, and Nitin.

Software Used Front End (UI) 
HTML
CSS
JavaScript
Bootstrap
Flask

Back End (Model for Face Recognition)
Python
Facenet
MTCNN
Haarcascade

Database
PostgreSQL

Project Outline and Specifications
Front End
There are 5 pages:
Welcome: Directs to user or admin login.
Admin Login: Allows registered admins to sign in. Admin data is securely hashed and encrypted.
Dashboard: Admins can manage users/classes, upload images, and view attendance. Image encodings are stored in the database.
User Login: Users capture their face via webcam for verification and attendance.

Back End
The backend uses Python, Facenet, MTCNN, and Haarcascade for facial recognition.
Admin Side: Uses MTCNN for facial landmark localization and Facenet to store embeddings in a 128-vector format.
User Side: Uses Haarcascade for fast face detection and Facenet to match face embeddings with the database.

Database
A PostgreSQL database with a schema of 8 tables, each with a unique 128-bit UUID primary key.
