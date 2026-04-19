# app.py - Main Streamlit Application with improved compatibility

import streamlit as st
import cv2
import pandas as pd
import pickle
import os
from datetime import datetime
import numpy as np
from PIL import Image
import tempfile
import sys

# Page configuration
st.set_page_config(
    page_title="Smart Attendance System",
    page_icon="📸",
    layout="wide"
)

# Title and description
st.title("📸 Smart Attendance System using Face Recognition")
st.markdown("---")

# Check Python version and show warning if needed
if sys.version_info >= (3, 12):
    st.warning("⚠️ You're using Python 3.12 or higher. Some packages might have compatibility issues. If you encounter errors, please use Python 3.10 or 3.11 for better compatibility.")

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

# Try to import face_recognition with error handling
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError as e:
    FACE_RECOGNITION_AVAILABLE = False
    st.error(f"Face recognition library not available: {str(e)}")
    st.info("Please make sure all dependencies are installed correctly. Try restarting the app.")

def train_model(images_folder):
    """Train the face recognition model"""
    if not FACE_RECOGNITION_AVAILABLE:
        st.error("Face recognition library is not available")
        return False, None, None
    
    try:
        student_encodings = []
        student_names = []
        
        # Get all image files
        image_files = [f for f in os.listdir(images_folder) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if len(image_files) == 0:
            st.error("No images found in the selected folder!")
            return False, None, None
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, filename in enumerate(image_files):
            status_text.text(f"Processing {filename}...")
            img_path = os.path.join(images_folder, filename)
            
            try:
                # Load and resize image
                image = face_recognition.load_image_file(img_path)
                
                # Resize for faster processing
                if image.shape[1] > 1000:  # If width > 1000 pixels
                    scale_factor = 1000 / image.shape[1]
                    new_width = int(image.shape[1] * scale_factor)
                    new_height = int(image.shape[0] * scale_factor)
                    image = cv2.resize(image, (new_width, new_height))
                
                # Get face encoding
                face_encodings = face_recognition.face_encodings(image)
                
                if len(face_encodings) > 0:
                    student_encodings.append(face_encodings[0])
                    student_names.append(os.path.splitext(filename)[0])
                    st.success(f"✓ Processed: {filename}")
                else:
                    st.warning(f"⚠ No face detected in {filename}")
            
            except Exception as e:
                st.error(f"Error processing {filename}: {str(e)}")
            
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
    if not FACE_RECOGNITION_AVAILABLE:
        st.error("Face recognition library is not available")
        return [], []
    
    try:
        # Convert PIL image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert BGR to RGB if needed (OpenCV uses BGR)
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Check if it's BGR (common from OpenCV)
            if image[0,0,0] > image[0,0,2]:  # Simple heuristic
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize for faster processing
        scale_factor = 0.5
        small_frame = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor)
        
        # Find face locations
        face_locations = face_recognition.face_locations(small_frame)
        
        if len(face_locations) == 0:
            return [], []
        
        # Scale back face locations
        face_locations = [(int(top/scale_factor), int(right/scale_factor), 
                          int(bottom/scale_factor), int(left/scale_factor)) 
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
        cv2.rectangle(img_copy, (left, bottom - 25), 
                     (left + label_bg[0] + 10, bottom), (0, 255, 0), -1)
        
        # Draw label text
        cv2.putText(img_copy, name, (left + 5, bottom - 8),
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
    
    if not FACE_RECOGNITION_AVAILABLE:
        st.error("⚠️ Face recognition library is not available. Please check the installation.")
        st.stop()
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_files = st.file_uploader(
            "Upload student images (jpg, jpeg, png)",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            help="Upload clear, front-facing photos of students"
        )
    
    with col2:
        folder_path = st.text_input("Or enter folder path (for deployed version):", 
                                   help="Example: /mount/src/your-repo/images")
    
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
    
    if not FACE_RECOGNITION_AVAILABLE:
        st.error("⚠️ Face recognition library is not available. Please check the installation.")
        st.stop()
    
    # Check if model is trained
    if not st.session_state.model_trained:
        # Try to load existing model
        if os.path.exists('trained_model.pickle'):
            try:
                with open('trained_model.pickle', 'rb') as f:
                    st.session_state.student_encodings, st.session_state.student_names = pickle.load(f)
                    st.session_state.model_trained = True
                    st.success("Existing model loaded successfully!")
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                st.warning("⚠ No valid trained model found! Please train the model first.")
                st.stop()
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
            
            if len(face_locations) > 0:
                # Draw boxes on image
                annotated_image = draw_boxes(image_np, recognized_names, face_locations)
                
                # Display annotated image
                with col2:
                    st.image(annotated_image, caption="Recognized Faces", use_column_width=True)
            else:
                st.warning("No faces detected in the image. Please try another photo.")
            
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
            
            if total_count > 0:
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
st.markdown("💡 **Tips for better accuracy:**")
st.markdown("""
- Use clear, front-facing photos for training
- Ensure good lighting in group photos
- Faces should be clearly visible and not too small
- For best results, use photos with faces looking directly at the camera
""")

# Add a requirements checker in sidebar
with st.sidebar.expander("ℹ️ System Info"):
    st.write(f"Python Version: {sys.version}")
    st.write(f"Face Recognition Available: {FACE_RECOGNITION_AVAILABLE}")
    if FACE_RECOGNITION_AVAILABLE:
        st.write("✅ All dependencies loaded successfully")
    else:
        st.write("❌ Some dependencies missing")
