from flask import Flask, render_template, request, Response, redirect, url_for, flash, jsonify,session
import psycopg2
from psycopg2 import sql, Error
import base64
import cv2 as cv
import numpy as np
from keras_facenet import FaceNet
import os
os.environ['TF'] = '2'
from datetime import datetime
import json
from mtcnn import MTCNN
from werkzeug.security import check_password_hash
from datetime import datetime, timedelta

app = Flask(__name__)
app.secret_key = 'nitins_secret_key'
# Database connection details
DB_HOST = "localhost"
DB_NAME = "face_recognition_attendance"
DB_USER = "postgres"
DB_PASS = "kai23"

def get_db_connection():
    conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASS)
    return conn

image = []
encode = []


def grab():
    try:
        u = []
        # Connect to the PostgreSQL database
        connection = psycopg2.connect(user="postgres",
                                      password="kai23",
                                      host="localhost",
                                      port="5432",
                                      database="face_recognition_attendance")

        cursor = connection.cursor()
        dd = []
        # Insert data into the something table
        cursor.execute("SELECT content FROM public.encoder")
        records = cursor.fetchall()
        for encodes in records:
            cursor.execute('''SELECT (user_id) FROM public.resources INNER JOIN public.encoder ON 
                           resources.id = encoder.resource_id WHERE encoder.content = (%s)''',(encodes,))
            d = cursor.fetchall()
            u.append(d[0])
            cursor.execute('''SELECT (name) FROM public.user WHERE id = (%s)''',(d[0],))
            d = cursor.fetchall()
            dd.append(d[0])
        return records, dd, u
    except (Exception, Error) as error:
        print("Error:", error)
        return None, None
    finally:
        # Close database connection
        if connection:
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")

cap  = cv.VideoCapture(0)

def get_frame():
    # Access the webcam
    while True:
        success, img = cap.read()
        if not success:
            break
        else:
            success, buffer1 = cv.imencode('.jpg', img)
            frame = buffer1.tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def extract_face(image):
        target_size = (160,160)
        detector = MTCNN()
        img = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        detection = detector.detect_faces(img)
        if detection:
           x, y, w, h = detection[0]['box']
           x, y = abs(x), abs(y)
           face = img[y: y + h, x: x + w]
           face_arr = cv.resize(face, target_size)
           return face_arr
        else:
            success = "False"
            message = "Face not detected!Retake"
            response = {"success": success, "message": message}
            return response
      

embedder = FaceNet()
cap = None
facenet = FaceNet()
haarcascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

def get_embedding(face_img):
    face_img = face_img.astype('float32')
    face_img = np.expand_dims(face_img, axis=0)
    embedding = embedder.embeddings(face_img)
    return embedding[0]

def verify(frame):
    tests, imagep, u = grab()
    encode = []
    print("hii")
    for x in tests:
        arr = x[0]
        b = np.fromstring(arr.strip('[]'), sep=' ')
        encode.append(b)
     
    # No image in the database
    if imagep == None:
        return "No user data avaliable"
    
    
    # Threshold for considering a face recognized
    threshold_distance = 0.8

    rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)
    

    if len(faces) == 0:
        success = "False"
        message = "Face not detected!Retake"
        response = {"success": success, "message": message, "image": None}
        return response
         

    for x, y, w, h in faces:
        # Extract face ROI
        face_img = rgb_img[y:y+h, x:x+w]
        face_img = cv.resize(face_img, (160, 160))
        face_img = np.expand_dims(face_img, axis=0)
        # face = extract_face(face_img)

        # Get embedding for the face
        face_embedding = facenet.embeddings(face_img)

        # Calculate distances between the face embedding and trained embeddings
        distances = np.linalg.norm(np.array(encode) - face_embedding, axis=1)
        # Find the index with minimum distance
        min_distance_index = np.argmin(distances)
        # Get the corresponding label and minimum distance
        min_distance = distances[min_distance_index]
        # If the minimum distance is below the threshold, recognize the face
        if min_distance < threshold_distance:
            predicted_label  = imagep[min_distance_index]
            print(u[min_distance_index])
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            name = str(predicted_label)
            name = name[2:len(name)-3]
            mark_attendance(u[min_distance_index][0])
            success = "True"
            message = f"{name}, you have been marked for attendance at {timestamp}."
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT content FROM resources WHERE user_id = %s", (u[min_distance_index][0],))
            image_data = cursor.fetchone()[0]  # Assuming there's only one image per user
            conn.close()

            # Encode the image data to base64
            encoded_image = base64.b64encode(image_data).decode('utf-8')

            response = {"success": success, "message": message, "image": encoded_image}
            return response

        else:
            success = "False"
            message = "No user data avaliable! Retake"
            response = {"success": success, "message": message, "image": None}
            return response


def mark_attendance(name):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Insert data into the table
        cursor.execute("INSERT INTO attendance (user_id) VALUES (%s)", (name,))

        # Commit changes and close connection
        conn.commit()
        cursor.close()
        conn.close()
        print("Row inserted successfully!")
    except (Exception, Error) as error:
        print("Error:", error)

def start_capture():
    global cap
    cap = cv.VideoCapture(0)

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/user_login')
def user_login():
    return render_template('user.html')

@app.route('/video_feed')
def video_feed():
    # feed()
    return Response(get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/admin_login')
def admin_login():
    return render_template('admin.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, hashed_password FROM admin WHERE username = %s",
        (username,)
    )
    admin = cursor.fetchone()
    
    if admin and check_password_hash(admin[1], password):
        session['admin_id'] = admin[0]
        cursor.execute(
            "INSERT INTO public.login_history (admin_id, login_time) VALUES (%s, CURRENT_TIMESTAMP)",
            (admin[0],)
        )
        conn.commit()
        cursor.close()
        conn.close()
        return redirect(url_for('dashboard'))
    else:
        cursor.close()
        conn.close()
        flash('Invalid username or password. Please try again.', 'danger')
        return redirect(url_for('admin_login'))
    
@app.route('/dashboard')
def dashboard():
    users = show_users()
    return render_template('dashboard.html', users=users)

def show_users():
    conn = get_db_connection()
    cursor = conn.cursor()

    # Fetch data from both tables using a JOIN
    query = """
        SELECT u.id, u.register_no, u.name, ct.class_name, r.content
        FROM public.user u
        LEFT JOIN public.user_class uc ON u.id = uc.user_id
        LEFT JOIN public.class_type ct ON uc.class_id = ct.id
        LEFT JOIN public.resources r ON u.id = r.user_id::uuid
    """
    cursor.execute(query)
    users = cursor.fetchall()

    users_with_images = []
    for user in users:
        user_id, register_no, name, class_name, content = user
        # Check if content (image) exists for the user
        if content:
            # Encode the image data as base64
            image_base64 = base64.b64encode(content).decode('utf-8')
            user_data = {
                "id": user_id,
                "register_no": register_no,
                "name": name,
                "class_name": class_name,
                "image_base64": image_base64
            }
        else:
            user_data = {
                "id": user_id,
                "register_no": register_no,
                "name": name,
                "class_name": class_name,
                "image_base64": None
            }
        users_with_images.append(user_data)

    cursor.close()
    conn.close()
    return users_with_images

@app.route('/add_user', methods=['POST'])
def add_user():
    try:
        data = request.get_json()
        
        register_no = data.get('register_no')
        name = data.get('name')
        class_name = data.get('class_name') 
        image_base64 = data.get('image')
        image_data = base64.b64decode(image_base64)
        admin_id = session.get('admin_id')

        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if register_no already exists
        cursor.execute(
            "SELECT COUNT(*) FROM public.user WHERE register_no = %s",
            (register_no,)
        )
        existing_user_count = cursor.fetchone()[0]
        if existing_user_count > 0:
            return jsonify({"error": "Register number already exists."}), 400

        cursor.execute(
            "INSERT INTO public.user (register_no, name, created_by, updated_by) VALUES (%s, %s, %s, %s) RETURNING id",
            (register_no, name, admin_id, admin_id)
        )
        user_id = cursor.fetchone()[0]
        cursor.execute(
            "SELECT id FROM public.class_type WHERE class_name = (%s)",
            (class_name,)
        )
        class_id = cursor.fetchone()[0]
        cursor.execute(
            "INSERT INTO public.user_class (class_id, user_id, created_by, updated_by) VALUES (%s, %s, %s, %s)",
            (class_id, user_id, admin_id, admin_id)
        )
        cursor.execute(
            "INSERT INTO resources (user_id, content, type, created_by, updated_by) VALUES (%s, %s, 'registered_user', %s, %s) RETURNING id",
            (user_id, psycopg2.Binary(image_data), admin_id, admin_id)
        )

        resource_id = cursor.fetchone()[0]
        np_array = np.frombuffer(image_data, dtype=np.uint8)
        image = cv.imdecode(np_array, cv.IMREAD_COLOR)

        face = extract_face(image)
        embedding = get_embedding(face)
        embedding_str = np.array_str(embedding)
        # print(type(embedding_str))
           
        cursor.execute("INSERT INTO public.encoder (resource_id, content, type, created_by, updated_by) VALUES (%s,%s,'array', %s, %s)",(resource_id, embedding_str, admin_id, admin_id))
        conn.commit()
        cursor.close()
        conn.close()

        print("User added successfully!")  # Debugging
        return jsonify({"message": "User added successfully!"}), 200

    except Exception as e:
        print(f"Error occurred: {e}")  # Debugging
        return jsonify({"message": "Failed to add user."}), 500

@app.route('/dashboard_data', methods=['GET'])
def dashboard_data():
    users = show_users()
    users_with_images = [(user[0], user[1], user[2]) for user in users] 
    return jsonify({"users": users_with_images})

@app.route('/update_user_data', methods=['POST'])
def update_user_data():
    try:
        # Extract updated user data from the request
        user_id = request.form['id']
        register_no = request.form['register_no']
        name = request.form['name']
        class_name = request.form['class_name']
        content = request.form.get('content')  # Get base64-encoded image content if available
        admin_id = session.get('admin_id')

        # Update the user table
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id FROM public.user WHERE register_no = %s",(register_no,)
        )

        user_id = cursor.fetchone()[0]
        cursor.execute(
            "UPDATE public.user SET register_no = %s, name = %s, updated_by = %s WHERE id = %s",
            (register_no, name, admin_id, user_id)
        )
        cursor.execute(
            "SELECT id FROM public.class_type WHERE class_name = (%s)",
            (class_name,)
        )

        print(class_name)
        class_id = cursor.fetchone()[0]
        print(class_id)
        print(user_id)
        cursor.execute(
            "UPDATE public.user_class SET class_id = %s, updated_by = %s WHERE user_id = %s",
            (class_id, admin_id, user_id)
        )
        print(user_id)

        # Update the resources table with the base64-encoded image content if available
        if content:
            # Decode the base64-encoded content
            image_data = base64.b64decode(content)
            cursor.execute(
                "UPDATE public.resources SET content = %s, updated_by = %s WHERE user_id = %s RETURNING id",
                (image_data, admin_id, user_id)
            )
            resource_id = cursor.fetchone()[0]
            print(resource_id)
            np_array = np.frombuffer(image_data, dtype=np.uint8)
            image = cv.imdecode(np_array, cv.IMREAD_COLOR)
            face = extract_face(image)
            embedding = get_embedding(face)
            embedding_str = np.array_str(embedding)
            cursor.execute("UPDATE public.encoder SET content = %s, updated_by = %s WHERE resource_id = %s",
                           (embedding_str, admin_id, resource_id))
        # Commit the changes
        conn.commit()
        cursor.close()
        conn.close()

        print("User data updated successfully!")  # Debugging
        return jsonify({"message": "User data updated successfully!"}), 200

    except Exception as e:
        print(f"Error occurred: {e}")  # Debugging
        return jsonify({"message": "Failed to update user data."}), 500
    
# Route to fetch user data by register number
@app.route('/get_user_data_by_register_no/<register_no>', methods=['GET'])
def get_user_data_by_register_no(register_no):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Fetch user data by register number
        cursor.execute(
            "SELECT u.id, u.register_no, u.name, r.content " +
            "FROM public.user u " +
            "JOIN public.resources r ON u.id = r.user_id " +
            "WHERE u.register_no = %s",
            (register_no,)
        )
        user_data = cursor.fetchone()
        print(user_data)
        
        
        cursor.execute(
            "SELECT ct.class_name " +
            "FROM public.class_type ct " +
            "JOIN public.user_class uc ON ct.id = uc.class_id " +
            "JOIN public.user u ON u.id = uc.user_id"
        )
        class_name = cursor.fetchone()
        print(class_name)
        cursor.close()
        conn.close()

        if user_data:
            # Convert binary data (content) to base64 encoding
            content_base64 = base64.b64encode(user_data[3]).decode('utf-8')
            print("hi")

            # Return the user data with content as base64 string
            return jsonify({
                'id': str(user_data[0]),
                'register_no': user_data[1],
                'name': user_data[2],
                'class': class_name,
                'content': content_base64
            })
        else:
            return jsonify({'message': 'User not found'}), 404

    except Exception as e:
        print(f"Error occurred: {e}")  # Debugging
        return jsonify({"message": "Failed to fetch user data."}), 500
    
# Route to delete user data
@app.route('/delete_user_data', methods=['POST'])
def delete_user_data():
    try:
        # Extract user ID from the request
        register_no = request.form['id']

        conn = get_db_connection()
        cursor = conn.cursor()

        # Delete user data from both tables
        cursor.execute(
            "DELETE FROM public.user WHERE register_no = %s RETURNING id",
            (register_no,)
        )
        user_id = cursor.fetchone()[0]
        cursor.execute(
            "DELETE FROM public.resources WHERE user_id = %s RETURNING id",
            (user_id,)
        )
        resource_id = cursor.fetchone()[0]
        cursor.execute(
            "DELETE FROM public.user_class WHERE user_id = %s",
            (user_id,)
        )
        cursor.execute(
            "DELETE FROM public.encoder WHERE resource_id = %s",
            (resource_id,)
        )   

        cursor.execute(
            "DELETE FROM public.attendance WHERE user_id = %s",
            (user_id,)
        )

        print(user_id)
        print(resource_id)
        print(register_no)

        conn.commit()
        cursor.close()
        conn.close()

        print("User data deleted successfully!")  # Debugging
        return jsonify({"message": "User data deleted successfully!"}), 200

    except Exception as e:
        print(f"Error occurred: {e}")  # Debugging
        return jsonify({"message": "Failed to delete user data."}), 500

@app.route('/classm')
def classm():
    conn = get_db_connection()
    cursor = conn.cursor()

    # Fetch classes from the database
    cursor.execute("SELECT id, class_name FROM class_type")
    classes = cursor.fetchall()

    classes_array = []
    for classs in classes:
        class_id, class_name = classs
        class_data = {
            "id": class_id,
            "class_name": class_name
        }
        classes_array.append(class_data)
    # print(classes_array)


    cursor.close()
    conn.close()
    return render_template('classm.html', classes=classes_array)

@app.route('/add_class', methods=['POST'])
def add_class():
    try:
        class_name = request.form['class_name']
        admin_id = session.get('admin_id')

        # Insert the new class into the database
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO class_type (class_name, created_by, updated_by) VALUES (%s, %s, %s)",
            (class_name, admin_id, admin_id)
        )
        conn.commit()
        cursor.close()
        conn.close()
        return jsonify({"class_name": class_name}), 200

    except Exception as e:
        print(f"Error occurred: {e}")  # Debugging
        return jsonify({"message": "Failed to add class."}), 500

@app.route('/update_class', methods=['POST'])
def update_class():
    try:
        class_id = request.form['id']
        class_name = request.form['class_name']
        admin_id = session.get('admin_id')

        # Update the class in the database
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE class_type SET class_name = %s, updated_by = %s WHERE id = %s",
            (class_name, admin_id, class_id)
        )
        conn.commit()

        cursor.close()
        conn.close()

        return jsonify({"message": "Class updated successfully!"}), 200

    except Exception as e:
        print(f"Error occurred: {e}")  # Debugging
        return jsonify({"message": "Failed to update class."}), 500

@app.route('/delete_class', methods=['POST'])
def delete_class():
    try:
        class_name = request.form['class_name']

        conn = get_db_connection()
        cursor = conn.cursor()

        # Check if there are any users in the class
        cursor.execute("""
            SELECT COUNT(*) 
            FROM public.user_class uc
            JOIN public.class_type ct ON uc.class_id = ct.id
            WHERE ct.class_name = %s
        """, (class_name,))
        user_count = cursor.fetchone()[0]
        if user_count > 0:
            cursor.close()  
            conn.close()
            return jsonify({"message": "Cannot delete class. There are users associated with this class."}), 400

        # Delete the class from the class_type table
        cursor.execute("""
            DELETE FROM public.class_type 
            WHERE class_name = %s
        """, (class_name,))

        conn.commit()
        cursor.close()
        conn.close()

        return jsonify({"message": "Class deleted successfully!"}), 200

    except Exception as e:
        print(f"Error occurred: {e}")  # Debugging
        return jsonify({"message": "Failed to delete class."}), 500

@app.route('/get_class_names')
def get_class_names():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT class_name FROM class_type")
        class_names = [row[0] for row in cursor.fetchall()]

        cursor.close()
        conn.close()

        return jsonify(class_names), 200
    except Exception as e:
        print(f"Error occurred: {e}")  # Debugging
        return jsonify({"message": "Failed to fetch class names."}), 500

    
@app.route('/get_user_attendance/<register_no>', methods=['GET'])
def get_user_attendance(register_no):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')

        query = """
            SELECT a.date
            FROM public.user u
            JOIN public.attendance a ON u.id = a.user_id
            WHERE u.register_no = %s
        """
        params = [register_no]

        if start_date and end_date:
            # Parse the end_date to ensure it's a date object and extend it to the end of the day
            end_date = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)
            query += " AND a.date >= %s AND a.date < %s"
            params.extend([start_date, end_date.strftime('%Y-%m-%d')])

        cursor.execute(query, params)
        attendance_records = cursor.fetchall()

        cursor.close()
        conn.close()

        if attendance_records:
            attendance_data = [
                {'date': record[0].strftime('%Y-%m-%d'), 'time': record[0].strftime('%H:%M:%S')}
                for record in attendance_records
            ]
            return jsonify({'success': True, 'attendance': attendance_data})
        else:
            return jsonify({'success': False, 'message': 'No attendance records found'}), 404

    except Exception as e:
        print(f"Error occurred: {e}")  # Debugging
        return jsonify({'success': False, 'message': 'Failed to fetch user attendance data.'}), 500
    
@app.route('/fetch_attendance_data')
def fetch_attendance_data():
    try:
        class_id = request.args.get('class_id')
        date = request.args.get('date')

        conn = get_db_connection()
        cursor = conn.cursor()

        # Query to get users and their attendance for the specified class and date
        cursor.execute("""
            SELECT register_no, name, attendance
            FROM (
                SELECT 
                    u.register_no, 
                    u.name, 
                    CASE WHEN a.user_id IS NOT NULL THEN TRUE ELSE FALSE END AS attendance,
                    ROW_NUMBER() OVER (PARTITION BY u.id ORDER BY a.date DESC) AS row_num
                FROM 
                    public.user u
                JOIN 
                    public.user_class uc ON u.id = uc.user_id
                LEFT JOIN 
                    public.attendance a ON u.id = a.user_id AND DATE(a.date) = DATE(%s)
                WHERE 
                    uc.class_id = %s
            ) AS subquery
            WHERE 
                row_num = 1;
        """, (date, class_id))
        attendance_data = cursor.fetchall()
        print("Attendance data:", attendance_data)
        cursor.close()
        conn.close()

        # Convert data to a list of dictionaries for easy JSON serialization
        attendance_list = [
            {'register_no': row[0], 'name': row[1], 'attendance': row[2]}
        for row in attendance_data]

        return jsonify(attendance_list)
    except Exception as e:
        print(f"Error occurred: {e}")  # Debugging
        return jsonify({'success': False, 'message': 'Failed to fetch attendance data.'}), 500
    
@app.route('/logout', methods=['POST'])
def logout():
    admin_id = session.get('admin_id')
    if admin_id:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE public.login_history SET signout_time = CURRENT_TIMESTAMP WHERE admin_id = %s AND signout_time IS NULL",
            (admin_id,)
        )
        conn.commit()
        cursor.close()
        conn.close()
        session.pop('admin_id', None)  # Remove admin_id from session
        return jsonify({"message": "Logged out successfully!"}), 200
    else:
        return jsonify({"message": "No active session found."}), 400
    
@app.route('/start_verification', methods=['POST'])
def start_verification():
    start_capture()
    #cam()
    return Response(get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/verify', methods=['POST'])
def verify_image_endpoint():
    file = request.files['image']
    image = cv.imdecode(np.fromstring(file.read(), np.uint8), cv.IMREAD_COLOR)
    result = verify(image)
    # print(result)
    return jsonify({"message": result})

@app.route('/stop_verification' , methods=['POST'])
def stop_verification():
    global cap
    if cap:
        cap.release()
        cap = None
    return 'Verification stopped', 200

if __name__ == '__main__':
    app.run(debug=True)