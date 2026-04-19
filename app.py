# app.py - Main Streamlit Application

import streamlit as st
import cv2
import face_recognition
import pandas as pd
import pickle
import os
from datetime import datetime
import numpy as np
from PIL import Image
import tempfile

# Page configuration
st.set_page_config(
    page_title="Smart Attendance System",
    page_icon="📸",
    layout="wide"
)

# Title and description
st.title("📸 Smart Attendance System using Face Recognition")
st.markdown("---")

# Sidebar for navigation
st.sidebar.title("Navigation")
option = st.sidebar.radio(
    "Choose an option",
    ["Train Model", "Take Attendance", "View Attendance Records"]
)

# Initialize session state for model
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'student_encodings' not in st.session_state:
    st.session_state.student_encodings = None
if 'student_names' not in st.session_state:
    st.session_state.student_names = None

def train_model(images_folder):
    """Train the face recognition model"""
    try:
        student_encodings = []
        student_names = []
        
        # Get all image files
        image_files = [f for f in os.listdir(images_folder) 
                      if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        if len(image_files) == 0:
            st.error("No images found in the selected folder!")
            return False, None, None
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, filename in enumerate(image_files):
            status_text.text(f"Processing {filename}...")
            img_path = os.path.join(images_folder, filename)
            
            # Load and resize image
            image = face_recognition.load_image_file(img_path)
            image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
            
            # Get face encoding
            face_encodings = face_recognition.face_encodings(image)
            
            if len(face_encodings) > 0:
                student_encodings.append(face_encodings[0])
                student_names.append(os.path.splitext(filename)[0])
                st.success(f"✓ Processed: {filename}")
            else:
                st.warning(f"⚠ No face detected in {filename}")
            
            progress_bar.progress((idx + 1) / len(image_files))
        
        status_text.text("Training completed!")
        
        if len(student_encodings) == 0:
            st.error("No faces were detected in any images!")
            return False, None, None
        
        # Save model
        with open('trained_model.pickle', 'wb') as f:
            pickle.dump((student_encodings, student_names), f)
        
        return True, student_encodings, student_names
    
    except Exception as e:
        st.error(f"Error during training: {str(e)}")
        return False, None, None

def recognize_faces(image, student_encodings, student_names, tolerance=0.6):
    """Recognize faces in the uploaded image"""
    try:
        # Convert PIL image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Resize for faster processing
        small_frame = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
        
        # Find face locations
        face_locations = face_recognition.face_locations(small_frame)
        
        if len(face_locations) == 0:
            return [], []
        
        # Scale back face locations
        face_locations = [(top*2, right*2, bottom*2, left*2) 
                         for top, right, bottom, left in face_locations]
        
        # Get face encodings
        face_encodings = face_recognition.face_encodings(image, face_locations)
        
        recognized_names = []
        recognized_locations = []
        
        for encoding, location in zip(face_encodings, face_locations):
            # Compare with known faces
            matches = face_recognition.compare_faces(
                student_encodings, encoding, tolerance=tolerance
            )
            face_distances = face_recognition.face_distance(student_encodings, encoding)
            
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    recognized_names.append(student_names[best_match_index])
                    recognized_locations.append(location)
                else:
                    recognized_names.append("Unknown")
                    recognized_locations.append(location)
            else:
                recognized_names.append("Unknown")
                recognized_locations.append(location)
        
        return recognized_names, recognized_locations
    
    except Exception as e:
        st.error(f"Error during face recognition: {str(e)}")
        return [], []

def draw_boxes(image, names, locations):
    """Draw bounding boxes and names on the image"""
    img_copy = image.copy()
    
    for name, (top, right, bottom, left) in zip(names, locations):
        # Draw rectangle
        cv2.rectangle(img_copy, (left, top), (right, bottom), (0, 255, 0), 2)
        
        # Draw label background
        label_bg = cv2.getTextSize(name, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)[0]
        cv2.rectangle(img_copy, (left, bottom - 20), 
                     (left + label_bg[0], bottom), (0, 255, 0), -1)
        
        # Draw label text
        cv2.putText(img_copy, name, (left, bottom - 5),
                   cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
    
    return img_copy

def mark_attendance(student_names, recognized_names):
    """Generate attendance sheet"""
    now = datetime.now()
    date = now.strftime('%d-%m-%Y')
    time = now.strftime('%H:%M:%S')
    
    attendance = pd.DataFrame({
        'Student Name': student_names,
        'Date': date,
        'Time': time,
        'Status': ['Present' if name in recognized_names else 'Absent' 
                  for name in student_names]
    })
    
    return attendance

# Training Section
if option == "Train Model":
    st.header("🎓 Train Face Recognition Model")
    st.markdown("Upload student images or specify folder path")
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_files = st.file_uploader(
            "Upload student images (jpg, jpeg, png)",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True
        )
    
    with col2:
        folder_path = st.text_input("Or enter Google Drive/Colab folder path:")
    
    if st.button("🚀 Train Model", type="primary"):
        if uploaded_files:
            # Create temporary directory for uploaded files
            with tempfile.TemporaryDirectory() as temp_dir:
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, 'wb') as f:
                        f.write(uploaded_file.getbuffer())
                
                success, encodings, names = train_model(temp_dir)
                if success:
                    st.session_state.model_trained = True
                    st.session_state.student_encodings = encodings
                    st.session_state.student_names = names
                    st.success(f"✅ Model trained successfully with {len(names)} students!")
                    st.balloons()
                    
                    # Display trained students
                    st.subheader("Trained Students:")
                    st.write(", ".join(names))
        
        elif folder_path and os.path.exists(folder_path):
            success, encodings, names = train_model(folder_path)
            if success:
                st.session_state.model_trained = True
                st.session_state.student_encodings = encodings
                st.session_state.student_names = names
                st.success(f"✅ Model trained successfully with {len(names)} students!")
                st.balloons()
                
                # Display trained students
                st.subheader("Trained Students:")
                st.write(", ".join(names))
        else:
            st.error("Please upload images or provide a valid folder path!")

# Attendance Section
elif option == "Take Attendance":
    st.header("📸 Take Attendance")
    
    # Check if model is trained
    if not st.session_state.model_trained:
        # Try to load existing model
        if os.path.exists('trained_model.pickle'):
            with open('trained_model.pickle', 'rb') as f:
                st.session_state.student_encodings, st.session_state.student_names = pickle.load(f)
                st.session_state.model_trained = True
                st.success("Existing model loaded successfully!")
        else:
            st.warning("⚠ No trained model found! Please train the model first.")
            st.stop()
    
    # Display trained students count
    st.info(f"📊 Model loaded with {len(st.session_state.student_names)} students")
    
    # Image upload for attendance
    uploaded_image = st.file_uploader(
        "Upload group photo for attendance",
        type=['jpg', 'jpeg', 'png']
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if uploaded_image:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if uploaded_image and st.button("🎯 Recognize Faces & Mark Attendance", type="primary"):
        with st.spinner("Processing image and recognizing faces..."):
            # Convert PIL image to numpy array
            image_np = np.array(image)
            
            # Recognize faces
            recognized_names, face_locations = recognize_faces(
                image_np,
                st.session_state.student_encodings,
                st.session_state.student_names
            )
            
            # Remove duplicates
            recognized_names_unique = list(set(recognized_names))
            
            # Filter out "Unknown"
            recognized_names_unique = [name for name in recognized_names_unique if name != "Unknown"]
            
            # Draw boxes on image
            annotated_image = draw_boxes(image_np, recognized_names, face_locations)
            
            # Display annotated image
            with col2:
                st.image(annotated_image, caption="Recognized Faces", use_column_width=True)
            
            # Generate attendance
            attendance_df = mark_attendance(
                st.session_state.student_names,
                recognized_names_unique
            )
            
            # Display attendance
            st.subheader("📋 Attendance Record")
            st.dataframe(attendance_df, use_container_width=True)
            
            # Statistics
            present_count = len(recognized_names_unique)
            total_count = len(st.session_state.student_names)
            absent_count = total_count - present_count
            
            col3, col4, col5 = st.columns(3)
            with col3:
                st.metric("✅ Present", present_count)
            with col4:
                st.metric("❌ Absent", absent_count)
            with col5:
                st.metric("📊 Attendance %", f"{(present_count/total_count)*100:.1f}%")
            
            # Save attendance to CSV
            csv_filename = f"attendance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            attendance_df.to_csv(csv_filename, index=False)
            
            # Download button
            with open(csv_filename, 'rb') as f:
                st.download_button(
                    label="📥 Download Attendance CSV",
                    data=f,
                    file_name=csv_filename,
                    mime="text/csv"
                )

# View Records Section
elif option == "View Attendance Records":
    st.header("📊 View Attendance Records")
    
    # List all CSV files in current directory
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv') and f.startswith('attendance_')]
    
    if csv_files:
        selected_file = st.selectbox("Select attendance record to view:", csv_files)
        
        if selected_file:
            df = pd.read_csv(selected_file)
            st.dataframe(df, use_container_width=True)
            
            # Display statistics
            st.subheader("Statistics")
            col1, col2 = st.columns(2)
            
            with col1:
                present_count = len(df[df['Status'] == 'Present'])
                st.metric("Present Students", present_count)
            
            with col2:
                absent_count = len(df[df['Status'] == 'Absent'])
                st.metric("Absent Students", absent_count)
            
            # Download button for selected file
            with open(selected_file, 'rb') as f:
                st.download_button(
                    label="📥 Download Selected CSV",
                    data=f,
                    file_name=selected_file,
                    mime="text/csv"
                )
    else:
        st.info("No attendance records found. Please take attendance first.")

# Footer
st.markdown("---")
st.markdown("💡 **Tip:** Make sure student images are clear and well-lit for better recognition accuracy.")
