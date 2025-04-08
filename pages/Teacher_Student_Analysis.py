import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Student Analysis",
    page_icon="👨‍🏫",
    layout="wide"
)

# Check if teacher is logged in
if "teacher_id" not in st.session_state:
    st.warning("You need to log in first!")
    if st.button("Go to Login Page"):
        st.switch_page("pages/Teacher_Login.py")
else:
    # Teacher is logged in, show the analysis interface
    teacher_display_name = st.session_state.get("teacher_name", st.session_state.teacher_id)
    st.title(f"Welcome, {teacher_display_name}")
    st.subheader("Student Performance Analysis")
    
    # Load the student data
    data_path = "data/Students_Grading_Dataset.csv"
    df = None

    if os.path.exists(data_path):
        try:
            # Try multiple encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'ISO-8859-1']
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(data_path, encoding=encoding)
                    break  # If successful, exit the loop
                except UnicodeDecodeError:
                    continue  # Try the next encoding
            
            if df is None:
                st.error("Unable to read dataset with any of the attempted encodings.")
            else:
                # Create a combined name column for display purposes
                df['Student_Name'] = df['First_Name'] + ' ' + df['Last_Name']
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            df = None
    else:
        st.error(f"Data file not found at {data_path}")
        df = None
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Select Student")
        
        # Student selection method
        selection_method = st.radio(
            "Choose selection method:",
            ["By Student ID", "By Name", "View All Students"]
        )
        
        if selection_method == "By Student ID" and df is not None:
            student_id = st.text_input("Enter Student ID")
            if student_id:
                student_data = df[df['Student_ID'] == student_id]
                if not student_data.empty:
                    st.success(f"Found student: {student_data.iloc[0]['Student_Name']}")
                else:
                    st.error("Student not found!")
        
        elif selection_method == "By Name" and df is not None:
            student_name = st.text_input("Enter Student Name")
            if student_name:
                student_data = df[df['Student_Name'].str.contains(student_name, case=False)]
                if not student_data.empty:
                    st.success(f"Found {len(student_data)} matching students")
                else:
                    st.error("No students found with that name!")
        
        elif df is not None:  # View All Students
            student_data = df
            st.success(f"Showing all {len(df)} students")
        
        # Display student list if we have data
        if df is not None and 'student_data' in locals() and not student_data.empty:
            st.write("Select a student to analyze:")
            
            # If too many students, limit display
            max_display = 20
            if len(student_data) > max_display and selection_method == "View All Students":
                st.warning(f"Showing first {max_display} of {len(student_data)} students. Please use search to narrow results.")
                display_data = student_data.head(max_display)
            else:
                display_data = student_data
                
            # Create a container for scrollable content
            for i, (_, student) in enumerate(display_data.iterrows()):
                key = f"student_button_{i}"
                if st.button(f"{student['Student_Name']} (ID: {student['Student_ID']})", key=key):
                    st.session_state.selected_student = student
                    st.rerun()
    
    with col2:
        if 'selected_student' in st.session_state and df is not None:
            try:
                student = st.session_state.selected_student
                st.subheader(f"Analysis for {student['Student_Name']}")
                
                # Create tabs for different analysis views
                tab1, tab2 = st.tabs(["Current Performance", "Recommendations"])
                
                with tab1:
                    # Current Performance Analysis
                    st.write("### Current Academic Performance")
                    
                    # Create metrics for key performance indicators
                    col_a, col_b, col_c = st.columns(3)
                    
                    with col_a:
                        try:
                            attendance_value = f"{float(student['Attendance']):.1f}%" if 'Attendance' in student and not pd.isna(student['Attendance']) else "N/A"
                            study_hours_value = f"{float(student['Study_Hours_per_Week']):.1f}" if 'Study_Hours_per_Week' in student and not pd.isna(student['Study_Hours_per_Week']) else "N/A"
                        except (KeyError, TypeError, ValueError):
                            attendance_value, study_hours_value = "N/A", "N/A"
                            
                        st.metric("Attendance", attendance_value)
                        st.metric("Study Hours/Week", study_hours_value)
                    with col_b:
                        try:
                            assignments_value = f"{float(student['Assignments_Avg']):.1f}" if 'Assignments_Avg' in student and not pd.isna(student['Assignments_Avg']) else "N/A"
                            quizzes_value = f"{float(student['Quizzes_Avg']):.1f}" if 'Quizzes_Avg' in student and not pd.isna(student['Quizzes_Avg']) else "N/A"
                        except (KeyError, TypeError, ValueError):
                            assignments_value, quizzes_value = "N/A", "N/A"
                            
                        st.metric("Assignments Avg", assignments_value)
                        st.metric("Quizzes Avg", quizzes_value)
                    with col_c:
                        try:
                            midterm_value = f"{float(student['Midterm_Score']):.1f}" if 'Midterm_Score' in student and not pd.isna(student['Midterm_Score']) else "N/A"
                            final_value = f"{float(student['Final_Score']):.1f}" if 'Final_Score' in student and not pd.isna(student['Final_Score']) else "N/A"
                        except (KeyError, TypeError, ValueError):
                            midterm_value, final_value = "N/A", "N/A"
                            
                        st.metric("Midterm Score", midterm_value)
                        st.metric("Final Score", final_value)
                    
                    # Performance visualization
                    st.write("### Performance Breakdown")
                    
                    # Create a radar chart for performance factors
                    categories = ['Attendance', 'Study Hours', 'Assignments', 'Quizzes', 'Midterm', 'Final']
                    
                    # Safely extract values with error handling
                    try:
                        attendance_value = float(student['Attendance']) if 'Attendance' in student and not pd.isna(student['Attendance']) else 0
                        study_hours_value = float(student['Study_Hours_per_Week']) if 'Study_Hours_per_Week' in student and not pd.isna(student['Study_Hours_per_Week']) else 0
                        assignments_value = float(student['Assignments_Avg']) if 'Assignments_Avg' in student and not pd.isna(student['Assignments_Avg']) else 0
                        quizzes_value = float(student['Quizzes_Avg']) if 'Quizzes_Avg' in student and not pd.isna(student['Quizzes_Avg']) else 0
                        midterm_value = float(student['Midterm_Score']) if 'Midterm_Score' in student and not pd.isna(student['Midterm_Score']) else 0
                        final_value = float(student['Final_Score']) if 'Final_Score' in student and not pd.isna(student['Final_Score']) else 0
                    except (KeyError, TypeError, ValueError):
                        # Default values if extraction fails
                        attendance_value, study_hours_value, assignments_value, quizzes_value, midterm_value, final_value = 0, 0, 0, 0, 0, 0
                    
                    # Normalize values with safety checks
                    values = [
                        min(1.0, max(0.0, attendance_value / 100)) if 100 != 0 else 0,
                        min(1.0, max(0.0, study_hours_value / 30)) if 30 != 0 else 0,
                        min(1.0, max(0.0, assignments_value / 100)) if 100 != 0 else 0,
                        min(1.0, max(0.0, quizzes_value / 100)) if 100 != 0 else 0,
                        min(1.0, max(0.0, midterm_value / 100)) if 100 != 0 else 0,
                        min(1.0, max(0.0, final_value / 100)) if 100 != 0 else 0
                    ]
                    
                    # Create radar chart
                    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
                    values = np.concatenate((values, [values[0]]))  # complete the polygon
                    angles = np.concatenate((angles, [angles[0]]))  # complete the polygon
                    
                    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
                    ax.plot(angles, values)
                    ax.fill(angles, values, alpha=0.25)
                    ax.set_xticks(angles[:-1])
                    ax.set_xticklabels(categories)
                    ax.set_ylim(0, 1)
                    st.pyplot(fig)
                
                with tab2:
                    # Recommendations
                    st.write("### Recommendations for Improvement")
                    
                    # Analyze current performance
                    try:
                        performance_factors = {
                            "Attendance": min(1.0, max(0.0, float(student['Attendance']) / 100)) if 'Attendance' in student and not pd.isna(student['Attendance']) else 0,
                            "Study Hours": min(1.0, max(0.0, float(student['Study_Hours_per_Week']) / 30)) if 'Study_Hours_per_Week' in student and not pd.isna(student['Study_Hours_per_Week']) else 0,
                            "Assignments": min(1.0, max(0.0, float(student['Assignments_Avg']) / 100)) if 'Assignments_Avg' in student and not pd.isna(student['Assignments_Avg']) else 0,
                            "Quizzes": min(1.0, max(0.0, float(student['Quizzes_Avg']) / 100)) if 'Quizzes_Avg' in student and not pd.isna(student['Quizzes_Avg']) else 0,
                            "Midterm": min(1.0, max(0.0, float(student['Midterm_Score']) / 100)) if 'Midterm_Score' in student and not pd.isna(student['Midterm_Score']) else 0,
                            "Final": min(1.0, max(0.0, float(student['Final_Score']) / 100)) if 'Final_Score' in student and not pd.isna(student['Final_Score']) else 0
                        }
                    except (KeyError, TypeError, ValueError):
                        # Default values if extraction fails
                        performance_factors = {
                            "Attendance": 0,
                            "Study Hours": 0,
                            "Assignments": 0,
                            "Quizzes": 0,
                            "Midterm": 0,
                            "Final": 0
                        }
                    
                    # Sort factors by performance (lowest to highest)
                    sorted_factors = sorted(performance_factors.items(), key=lambda x: x[1])
                    lowest_factors = sorted_factors[:3]  # get 3 lowest performing factors
                    
                    # Generate recommendations based on lowest performing factors
                    st.write("#### Focus Areas")
                    for factor, value in lowest_factors:
                        st.write(f"• **{factor}**: Current performance at {value:.1%} of optimal")
                    
                    st.write("#### Specific Recommendations")
                    for factor, value in lowest_factors:
                        if factor == "Attendance":
                            st.write("• **Attendance Improvement**:")
                            st.write("  - Set up regular attendance tracking")
                            st.write("  - Schedule regular check-ins")
                            st.write("  - Consider implementing an attendance improvement plan")
                        
                        elif factor == "Study Hours":
                            st.write("• **Study Time Enhancement**:")
                            st.write("  - Create a structured study schedule")
                            st.write("  - Set specific study goals for each session")
                            st.write("  - Consider forming study groups")
                        
                        elif factor == "Assignments":
                            st.write("• **Assignment Performance**:")
                            st.write("  - Review assignment submission patterns")
                            st.write("  - Provide additional resources for challenging topics")
                            st.write("  - Consider assignment extensions if needed")
                        
                        elif factor == "Quizzes":
                            st.write("• **Quiz Performance**:")
                            st.write("  - Implement regular quiz practice sessions")
                            st.write("  - Review quiz-taking strategies")
                            st.write("  - Provide additional practice materials")
                        
                        elif factor == "Midterm":
                            st.write("• **Midterm Exam Preparation**:")
                            st.write("  - Create a comprehensive study plan")
                            st.write("  - Schedule regular review sessions")
                            st.write("  - Provide practice exams")
                        
                        elif factor == "Final":
                            st.write("• **Final Exam Strategy**:")
                            st.write("  - Develop a long-term preparation plan")
                            st.write("  - Schedule regular progress reviews")
                            st.write("  - Create a comprehensive study guide")
                    
                    # Additional recommendations based on other factors
                    st.write("#### Additional Considerations")
                    try:
                        stress_level = float(student['Stress_Level (1-10)']) if 'Stress_Level (1-10)' in student and not pd.isna(student['Stress_Level (1-10)']) else 5
                        sleep_hours = float(student['Sleep_Hours_per_Night']) if 'Sleep_Hours_per_Night' in student and not pd.isna(student['Sleep_Hours_per_Night']) else 7
                        extracurricular = student['Extracurricular_Activities'] if 'Extracurricular_Activities' in student and not pd.isna(student['Extracurricular_Activities']) else 'No'
                    except (KeyError, TypeError, ValueError):
                        stress_level, sleep_hours, extracurricular = 5, 7, 'No'
                    
                    if stress_level > 7:
                        st.write("• **Stress Management**:")
                        st.write("  - Implement stress reduction techniques")
                        st.write("  - Consider counseling services")
                        st.write("  - Adjust workload if necessary")
                    
                    if sleep_hours < 7:
                        st.write("• **Sleep Schedule**:")
                        st.write("  - Establish a regular sleep schedule")
                        st.write("  - Create a bedtime routine")
                        st.write("  - Avoid late-night studying")
                    
                    if extracurricular == 'No':
                        st.write("• **Extracurricular Engagement**:")
                        st.write("  - Encourage participation in academic clubs")
                        st.write("  - Suggest relevant extracurricular activities")
                        st.write("  - Balance academic and non-academic activities")
            except Exception as e:
                st.error(f"Error displaying student data: {str(e)}")
                st.info("Try selecting a different student or refresh the page.")
    
    # Footer with navigation
    st.markdown("---")
    col_nav1, col_nav2 = st.columns(2)
    
    with col_nav1:
        if st.button("Log Out"):
            # Clear session state
            for key in ["teacher_id", "teacher_name", "selected_student"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.switch_page("Home.py")
    
    with col_nav2:
        if st.button("Return to Home"):
            st.switch_page("Home.py") 