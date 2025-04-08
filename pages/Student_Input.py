import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime

# Try to import scikit-learn modules, but handle import errors gracefully
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn.pipeline import make_pipeline
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    st.warning("Scikit-learn could not be imported. Machine learning features will be limited.")

# Page configuration
st.set_page_config(
    page_title="Student Input",
    page_icon="👨‍🎓",
    layout="wide"
)

# Check if user is logged in
if "student_id" not in st.session_state or "student_name" not in st.session_state:
    st.warning("You need to log in first!")
    if st.button("Go to Login Page"):
        st.switch_page("pages/Student_Login.py")
else:
    # User is logged in, show the student input form
    st.title(f"Welcome, {st.session_state.student_name}")
    st.subheader("Input your data to predict your future performance")
    
    # Try to pre-fill some data based on student ID
    student_data = None
    df = None
    if os.path.exists("data/Students_Grading_Dataset.csv"):
        try:
            # Try multiple encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'ISO-8859-1']
            
            for encoding in encodings:
                try:
                    df = pd.read_csv("data/Students_Grading_Dataset.csv", encoding=encoding)
                    break  # If successful, exit the loop
                except UnicodeDecodeError:
                    continue  # Try the next encoding
            
            if df is None:
                st.error("Unable to read dataset with any of the attempted encodings.")
            else:
                # Check if required columns exist
                required_columns = ['Student_ID', 'First_Name', 'Last_Name', 'Age', 'Gender', 'Department']
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    st.warning(f"The dataset is missing some expected columns: {', '.join(missing_columns)}")
                
                try:
                    student_record = df[df['Student_ID'] == st.session_state.student_id]
                    if not student_record.empty:
                        student_data = student_record.iloc[0]
                    else:
                        st.warning(f"No data found for student ID: {st.session_state.student_id}")
                        student_data = None
                except Exception as e:
                    st.error(f"Error retrieving student data: {str(e)}")
                    student_data = None
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            df = None
    
    # Create a form for student inputs
    with st.form("student_form"):
        # Basic information display - non-editable
        st.subheader("Your Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write(f"**Student ID:** {st.session_state.student_id}")
            st.write(f"**Name:** {st.session_state.student_name}")
        
        with col2:
            if student_data is not None:
                st.write(f"**Age:** {student_data['Age']}")
                st.write(f"**Gender:** {student_data['Gender']}")
            
        with col3:
            if student_data is not None:
                st.write(f"**Department:** {student_data['Department']}")
        
        st.markdown("---")
        
        # Current academic performance - editable
        st.subheader("Current Academic Performance")
        col4, col5 = st.columns(2)
        
        with col4:
            try:
                attendance = st.slider("Attendance Percentage", min_value=0.0, max_value=100.0, 
                                 value=float(student_data['Attendance']) if student_data is not None and 'Attendance' in student_data and not pd.isna(student_data['Attendance']) else 80.0, step=0.1)
                study_hours = st.slider("Study Hours per Week", min_value=0.0, max_value=30.0, 
                                  value=float(student_data['Study_Hours_per_Week']) if student_data is not None and 'Study_Hours_per_Week' in student_data and not pd.isna(student_data['Study_Hours_per_Week']) else 15.0, step=0.5)
                assignments_avg = st.slider("Assignments Average Score", min_value=0.0, max_value=100.0, 
                                      value=float(student_data['Assignments_Avg']) if student_data is not None and 'Assignments_Avg' in student_data and not pd.isna(student_data['Assignments_Avg']) else 75.0, step=0.1)
            except (KeyError, TypeError, ValueError) as e:
                st.warning(f"Error loading student data: {str(e)}. Using default values.")
                attendance = 80.0
                study_hours = 15.0
                assignments_avg = 75.0
        
        with col5:
            try:
                quizzes_avg = st.slider("Quizzes Average Score", min_value=0.0, max_value=100.0, 
                                  value=float(student_data['Quizzes_Avg']) if student_data is not None and 'Quizzes_Avg' in student_data and not pd.isna(student_data['Quizzes_Avg']) else 75.0, step=0.1)
                participation = st.slider("Participation Score (0-10)", min_value=0.0, max_value=10.0, 
                                    value=float(student_data['Participation_Score']) if student_data is not None and 'Participation_Score' in student_data and not pd.isna(student_data['Participation_Score']) else 5.0, step=0.1)
                projects_score = st.slider("Projects Score", min_value=0.0, max_value=100.0, 
                                     value=float(student_data['Projects_Score']) if student_data is not None and 'Projects_Score' in student_data and not pd.isna(student_data['Projects_Score']) else 75.0, step=0.1)
            except (KeyError, TypeError, ValueError) as e:
                st.warning(f"Error loading student data: {str(e)}. Using default values.")
                quizzes_avg = 75.0
                participation = 5.0
                projects_score = 75.0
        
        st.markdown("---")
        
        # Exam expectations
        st.subheader("Expected Exam Performance")
        col6, col7 = st.columns(2)
        
        with col6:
            try:
                midterm_score = st.slider("Expected Midterm Score", min_value=0.0, max_value=100.0, 
                                    value=float(student_data['Midterm_Score']) if student_data is not None and 'Midterm_Score' in student_data and not pd.isna(student_data['Midterm_Score']) else 70.0, step=0.1)
            except (KeyError, TypeError, ValueError) as e:
                st.warning(f"Error loading student data: {str(e)}. Using default values.")
                midterm_score = 70.0
        
        with col7:
            try:
                final_score = st.slider("Expected Final Score", min_value=0.0, max_value=100.0, 
                                  value=float(student_data['Final_Score']) if student_data is not None and 'Final_Score' in student_data and not pd.isna(student_data['Final_Score']) else 70.0, step=0.1)
            except (KeyError, TypeError, ValueError) as e:
                st.warning(f"Error loading student data: {str(e)}. Using default values.")
                final_score = 70.0
        
        st.markdown("---")
        
        # Additional factors
        st.subheader("Additional Factors")
        col8, col9 = st.columns(2)
        
        with col8:
            try:
                extracurricular = st.selectbox("Extracurricular Activities", options=["Yes", "No"],
                                         index=0 if student_data is not None and 'Extracurricular_Activities' in student_data and student_data['Extracurricular_Activities'] == 'Yes' else 1)
                
                stress_level = st.slider("Stress Level (1-10)", min_value=1, max_value=10, 
                                    value=int(student_data['Stress_Level (1-10)']) if student_data is not None and 'Stress_Level (1-10)' in student_data and not pd.isna(student_data['Stress_Level (1-10)']) else 5)
            except (KeyError, TypeError, ValueError) as e:
                st.warning(f"Error loading student data: {str(e)}. Using default values.")
                extracurricular = "No"
                stress_level = 5
        
        with col9:
            try:
                sleep_hours = st.slider("Sleep Hours per Night", min_value=4.0, max_value=10.0, 
                                  value=float(student_data['Sleep_Hours_per_Night']) if student_data is not None and 'Sleep_Hours_per_Night' in student_data and not pd.isna(student_data['Sleep_Hours_per_Night']) else 7.0, step=0.1)
                
                internet_access = st.selectbox("Internet Access at Home", options=["Yes", "No"],
                                         index=0 if student_data is not None and 'Internet_Access_at_Home' in student_data and student_data['Internet_Access_at_Home'] == 'Yes' else 1)
            except (KeyError, TypeError, ValueError) as e:
                st.warning(f"Error loading student data: {str(e)}. Using default values.")
                sleep_hours = 7.0
                internet_access = "Yes"
        
        # Prediction model selection
        st.markdown("---")
        st.subheader("Prediction Model")
        prediction_model = st.selectbox(
            "Select Prediction Model",
            ["Basic Weighted Formula", "Random Forest ML Model", "Ensemble (Combined Model)"]
        )
        
        # Submit button
        submitted = st.form_submit_button("Predict My Performance")
    
    # Process the form and make predictions when submitted
    if submitted:
        # Store current data for prediction calculation
        current_data = {
            "attendance": attendance,
            "study_hours": study_hours,
            "assignments_avg": assignments_avg,
            "quizzes_avg": quizzes_avg,
            "participation": participation,
            "projects_score": projects_score,
            "midterm_score": midterm_score,
            "final_score": final_score,
            "extracurricular": extracurricular,
            "stress_level": stress_level,
            "sleep_hours": sleep_hours,
            "internet_access": internet_access
        }
        
        # Calculate normalized scores (0-1 scale) for each factor
        attendance_norm = attendance / 100
        study_norm = study_hours / 30
        assignments_norm = assignments_avg / 100
        quizzes_norm = quizzes_avg / 100
        participation_norm = participation / 10
        projects_norm = projects_score / 100
        midterm_norm = midterm_score / 100
        final_norm = final_score / 100
        
        # Sleep hours has an optimal range (7-9 hours), convert to 0-1 scale
        if 7 <= sleep_hours <= 9:
            sleep_norm = 1.0
        else:
            sleep_norm = 1.0 - min(abs(sleep_hours - 8), 4) / 4
        
        # Stress has negative impact, invert the scale (higher stress = lower score)
        stress_norm = 1.0 - (stress_level - 1) / 9
        
        # Extracurricular activities and internet access impact
        extra_norm = 1.0 if extracurricular == "Yes" else 0.8
        internet_norm = 1.0 if internet_access == "Yes" else 0.9
        
        # Basic weighted prediction model
        basic_predicted_score = (
            midterm_norm * 100 * 0.15 +
            final_norm * 100 * 0.20 +
            assignments_norm * 100 * 0.15 +
            quizzes_norm * 100 * 0.10 +
            participation_norm * 10 * 0.05 +
            projects_norm * 100 * 0.15 +
            attendance_norm * 100 * 0.10 +
            study_norm * 100 * 0.05 +
            sleep_norm * 100 * 0.02 +
            stress_norm * 100 * 0.02 +
            extra_norm * 5 +
            internet_norm * 5
        )
        
        # ML-based prediction using Random Forest
        ml_predicted_score = basic_predicted_score  # Default fallback
        model_accuracy = None
        feature_importance = None
        
        if SKLEARN_AVAILABLE and df is not None and len(df) > 20:  # Only use ML if scikit-learn is available and we have enough data
            try:
                # Prepare the data for ML model
                # Convert categorical features to numeric
                df_ml = df.copy()
                
                # Check if required columns exist for categorical mapping
                categorical_columns = {
                    'Extracurricular_Activities': {'Yes': 1, 'No': 0},
                    'Internet_Access_at_Home': {'Yes': 1, 'No': 0},
                    'Gender': {'Male': 0, 'Female': 1}
                }
                
                for col, mapping in categorical_columns.items():
                    if col in df_ml.columns:
                        df_ml[col] = df_ml[col].map(mapping)
                
                # Fill missing values in numeric columns with median
                for col in df_ml.select_dtypes(include=[np.number]).columns:
                    if df_ml[col].isna().any():
                        df_ml[col] = df_ml[col].fillna(df_ml[col].median())
                
                # Select relevant features
                features = [
                    'Age', 'Attendance', 'Study_Hours_per_Week', 
                    'Midterm_Score', 'Final_Score', 'Assignments_Avg', 
                    'Quizzes_Avg', 'Participation_Score', 'Projects_Score',
                    'Stress_Level (1-10)', 'Sleep_Hours_per_Night'
                ]
                
                # Add binary columns
                if 'Extracurricular_Activities' in df_ml.columns:
                    features.append('Extracurricular_Activities')
                if 'Internet_Access_at_Home' in df_ml.columns:
                    features.append('Internet_Access_at_Home')
                
                # Select only features that exist in the DataFrame
                features = [f for f in features if f in df_ml.columns]
                
                # Check if we have enough features
                if len(features) < 3:
                    st.warning("Not enough features available for ML prediction. Falling back to basic formula.")
                    ml_predicted_score = basic_predicted_score
                    model_accuracy = None
                    feature_importance = None
                else:
                    # Check if Total_Score column exists
                    if 'Total_Score' not in df_ml.columns:
                        st.warning("Total_Score column not found in dataset. Falling back to basic formula.")
                        ml_predicted_score = basic_predicted_score
                        model_accuracy = None
                        feature_importance = None
                    else:
                        # Filter out rows with NaN values
                        df_clean = df_ml.dropna(subset=features + ['Total_Score'])
                        
                        if len(df_clean) > 20:  # Still enough data after cleaning
                            try:
                                X = df_clean[features]
                                y = df_clean['Total_Score']
                                
                                # Split data for training and testing
                                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                                
                                # Create a pipeline with scaling and Random Forest
                                model = make_pipeline(
                                    StandardScaler(),
                                    RandomForestRegressor(n_estimators=100, random_state=42)
                                )
                                
                                # Train the model
                                model.fit(X_train, y_train)
                                
                                # Evaluate model
                                y_pred = model.predict(X_test)
                                model_accuracy = 1 - np.sqrt(mean_squared_error(y_test, y_pred)) / 100
                                
                                # Get feature importance
                                rf_model = model.named_steps['randomforestregressor']
                                feature_importance = dict(zip(features, rf_model.feature_importances_))
                                
                                # Create a feature vector for the current student
                                try:
                                    student_age = float(student_data['Age']) if student_data is not None and 'Age' in student_data and not pd.isna(student_data['Age']) else 20
                                except (KeyError, TypeError, ValueError):
                                    student_age = 20
                                    
                                # Create a dictionary of student features
                                student_feature_dict = {
                                    'Age': student_age,
                                    'Attendance': attendance,
                                    'Study_Hours_per_Week': study_hours,
                                    'Midterm_Score': midterm_score,
                                    'Final_Score': final_score,
                                    'Assignments_Avg': assignments_avg,
                                    'Quizzes_Avg': quizzes_avg,
                                    'Participation_Score': participation,
                                    'Projects_Score': projects_score,
                                    'Stress_Level (1-10)': stress_level,
                                    'Sleep_Hours_per_Night': sleep_hours
                                }
                                
                                # Add binary features if they were in the model
                                if 'Extracurricular_Activities' in features:
                                    student_feature_dict['Extracurricular_Activities'] = 1 if extracurricular == "Yes" else 0
                                if 'Internet_Access_at_Home' in features:
                                    student_feature_dict['Internet_Access_at_Home'] = 1 if internet_access == "Yes" else 0
                                
                                # Create feature array in the same order as the model expects
                                student_features = np.array([student_feature_dict.get(f, 0) for f in features])
                                
                                # Make prediction
                                ml_predicted_score = model.predict(student_features.reshape(1, -1))[0]
                                
                                # Ensure the prediction is within a reasonable range
                                ml_predicted_score = max(0, min(100, ml_predicted_score))
                            except Exception as e:
                                st.warning(f"Error in ML model training or prediction: {str(e)}. Falling back to basic formula.")
                                ml_predicted_score = basic_predicted_score
                                model_accuracy = None
                                feature_importance = None
                        else:
                            st.warning("Not enough clean data for ML prediction. Falling back to basic formula.")
                            ml_predicted_score = basic_predicted_score
                            model_accuracy = None
                            feature_importance = None
            except Exception as e:
                st.warning(f"Error in machine learning prediction: {str(e)}. Falling back to basic formula.")
                ml_predicted_score = basic_predicted_score
                model_accuracy = None
                feature_importance = None
        
        # Choose the prediction based on selected model
        if prediction_model == "Basic Weighted Formula":
            predicted_score = basic_predicted_score
            model_description = "Basic weighted formula using factors with predefined weights"
        elif prediction_model == "Random Forest ML Model" and model_accuracy is not None:
            predicted_score = ml_predicted_score
            model_description = f"Machine learning model (Random Forest) with {model_accuracy:.2%} accuracy"
        else:  # Ensemble or fallback if ML model isn't available
            if model_accuracy is not None:
                # Ensemble approach: weighted average of both models
                predicted_score = basic_predicted_score * 0.4 + ml_predicted_score * 0.6
                model_description = "Ensemble model combining basic formula and machine learning prediction"
            else:
                predicted_score = basic_predicted_score
                model_description = "Basic weighted formula (ML model not available with current data)"
        
        # Calculate letter grade from predicted score
        if predicted_score >= 90:
            predicted_grade = "A"
        elif predicted_score >= 80:
            predicted_grade = "B"
        elif predicted_score >= 70:
            predicted_grade = "C"
        elif predicted_score >= 60:
            predicted_grade = "D"
        else:
            predicted_grade = "F"
        
        # Display prediction results
        st.markdown("---")
        st.header("Your Prediction Results")
        
        # Show which model was used
        st.info(f"**Prediction Model:** {model_description}")
        
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            # Score Prediction
            st.metric(
                label="Predicted Total Score", 
                value=f"{predicted_score:.2f}"
            )
            
            # Create a gauge chart for Score
            fig1, ax1 = plt.subplots(figsize=(8, 1))
            ax1.barh(0, 100, color='lightgray', height=0.3)
            ax1.barh(0, predicted_score, color='blue', height=0.3)
            ax1.set_xlim(0, 100)
            ax1.set_yticks([])
            ax1.set_xticks([0, 20, 40, 60, 80, 100])
            ax1.set_title('Predicted Score')
            st.pyplot(fig1)
            
            # Display predicted grade
            st.metric(
                label="Predicted Grade",
                value=predicted_grade
            )
            
            # If ML model was used, show comparison with basic model
            if prediction_model != "Basic Weighted Formula" and model_accuracy is not None:
                st.write("**Prediction Comparison:**")
                comparison_df = pd.DataFrame({
                    'Model': ['Basic Formula', 'Machine Learning', 'Selected Model'],
                    'Predicted Score': [f"{basic_predicted_score:.2f}", f"{ml_predicted_score:.2f}", f"{predicted_score:.2f}"]
                })
                st.dataframe(comparison_df)
        
        with col_res2:
            # Key improvement areas
            st.subheader("Key Areas for Improvement")
            
            # Initialize improvement_areas with default values
            improvement_areas = []
            
            # Identify the weakest areas - using feature importance if available
            if feature_importance is not None and prediction_model != "Basic Weighted Formula":
                try:
                    # Use the ML model's feature importance
                    # Normalize the current values to 0-1
                    current_values = {
                        "Attendance": attendance_norm,
                        "Study Hours": study_norm,
                        "Assignments": assignments_norm,
                        "Quizzes": quizzes_norm,
                        "Participation": participation_norm,
                        "Projects": projects_norm,
                        "Midterm Exam": midterm_norm,
                        "Final Exam": final_norm,
                        "Sleep Quality": sleep_norm,
                        "Stress Management": stress_norm
                    }
                    
                    # Map feature names from the model to display names
                    feature_map = {
                        'Attendance': 'Attendance',
                        'Study_Hours_per_Week': 'Study Hours',
                        'Assignments_Avg': 'Assignments',
                        'Quizzes_Avg': 'Quizzes',
                        'Participation_Score': 'Participation',
                        'Projects_Score': 'Projects',
                        'Midterm_Score': 'Midterm Exam',
                        'Final_Score': 'Final Exam',
                        'Sleep_Hours_per_Night': 'Sleep Quality',
                        'Stress_Level (1-10)': 'Stress Management'
                    }
                    
                    # Combine importance with current values
                    weighted_improvement = {}
                    for feature, importance in feature_importance.items():
                        if feature in feature_map:
                            display_name = feature_map[feature]
                            if display_name in current_values:
                                # Lower values and higher importance = more potential for improvement
                                weighted_improvement[display_name] = importance * (1 - current_values[display_name])
                    
                    # Sort by weighted improvement potential (high to low)
                    sorted_improvement = sorted(weighted_improvement.items(), key=lambda x: x[1], reverse=True)
                    improvement_areas = sorted_improvement[:3]
                    
                    # Display improvement areas with ML-derived importance
                    st.write("Based on machine learning analysis, focus on these areas:")
                    for area, score in improvement_areas:
                        perf_value = current_values[area] * 100
                        importance = score / max(weighted_improvement.values()) * 100 if weighted_improvement else 0
                        st.write(f"• **{area}**: Current performance {perf_value:.1f}%, Impact potential {importance:.1f}%")
                except Exception as e:
                    st.warning(f"Error calculating improvement areas: {str(e)}. Using basic formula instead.")
                    # Fall back to basic formula
                    factors = {
                        "Attendance": attendance_norm,
                        "Study Hours": study_norm,
                        "Assignments": assignments_norm,
                        "Quizzes": quizzes_norm,
                        "Participation": participation_norm,
                        "Projects": projects_norm,
                        "Midterm Exam": midterm_norm,
                        "Final Exam": final_norm,
                        "Sleep Quality": sleep_norm,
                        "Stress Management": stress_norm
                    }
                    
                    # Sort factors by value and get the 3 lowest
                    sorted_factors = sorted(factors.items(), key=lambda x: x[1])
                    lowest_factors = sorted_factors[:3]
                    
                    # Define improvement_areas for basic formula to use in recommendations
                    improvement_areas = lowest_factors
                    
                    for factor, value in lowest_factors:
                        st.write(f"• **{factor}**: {value*100:.1f}% of optimal")
            else:
                # Use the basic model's assessment
                try:
                    factors = {
                        "Attendance": attendance_norm,
                        "Study Hours": study_norm,
                        "Assignments": assignments_norm,
                        "Quizzes": quizzes_norm,
                        "Participation": participation_norm,
                        "Projects": projects_norm,
                        "Midterm Exam": midterm_norm,
                        "Final Exam": final_norm,
                        "Sleep Quality": sleep_norm,
                        "Stress Management": stress_norm
                    }
                    
                    # Sort factors by value and get the 3 lowest
                    sorted_factors = sorted(factors.items(), key=lambda x: x[1])
                    lowest_factors = sorted_factors[:3]
                    
                    # Define improvement_areas for basic formula to use in recommendations
                    improvement_areas = lowest_factors
                    
                    for factor, value in lowest_factors:
                        st.write(f"• **{factor}**: {value*100:.1f}% of optimal")
                except Exception as e:
                    st.warning(f"Error calculating improvement areas: {str(e)}")
                    st.write("• **General**: Focus on improving your overall academic performance")
                    improvement_areas = [("General", 0.5)]
            
            # General recommendations
            st.subheader("Recommendations")
            try:
                if predicted_score < 60:
                    st.error("Your predicted score indicates you may be at risk of failing.")
                    st.write("Focus on improving the following:")
                    if improvement_areas and "Attendance" in [f[0] for f in improvement_areas]:
                        st.write("- Improve your class attendance significantly")
                    if improvement_areas and "Study Hours" in [f[0] for f in improvement_areas]:
                        st.write("- Increase your weekly study hours")
                    if improvement_areas and ("Midterm Exam" in [f[0] for f in improvement_areas] or "Final Exam" in [f[0] for f in improvement_areas]):
                        st.write("- Get help with exam preparation through tutoring")
                elif predicted_score < 75:
                    st.warning("Your predicted score shows room for improvement.")
                    st.write("Consider working on:")
                    if improvement_areas:
                        for factor, _ in improvement_areas:
                            st.write(f"- Improving your {factor.lower()}")
                    else:
                        st.write("- Focus on your weakest academic areas")
                else:
                    st.success("Your predicted score looks good!")
                    st.write("To excel further:")
                    st.write("- Maintain your current performance")
                    st.write("- Consider setting higher goals in your strongest areas")
                    if stress_level > 7:
                        st.write("- Work on reducing your stress levels to prevent burnout")
            except Exception as e:
                st.warning(f"Error generating recommendations: {str(e)}")
                st.write("General recommendations:")
                st.write("- Focus on improving your weakest areas")
                st.write("- Maintain a consistent study schedule")
                st.write("- Seek help when needed")
        
        # Factors visualization
        st.subheader("Performance Factors Analysis")
        
        # Create a horizontal bar chart of all factors
        all_factors = {
            "Midterm Exam": midterm_norm,
            "Final Exam": final_norm,
            "Assignments": assignments_norm,
            "Quizzes": quizzes_norm,
            "Participation": participation_norm,
            "Projects": projects_norm,
            "Attendance": attendance_norm,
            "Study Time": study_norm,
            "Sleep Quality": sleep_norm,
            "Stress Management": stress_norm
        }
        
        # Create tabs for visualization
        viz_tab1, viz_tab2 = st.tabs(["Performance Level", "Model Insights"])
        
        with viz_tab1:
            # Sort by value
            sorted_all_factors = sorted(all_factors.items(), key=lambda x: x[1])
            factor_names = [f[0] for f in sorted_all_factors]
            factor_values = [f[1] for f in sorted_all_factors]
            
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            bars = ax2.barh(factor_names, factor_values, color='skyblue')
            ax2.set_xlim(0, 1)
            ax2.set_xlabel('Performance Level (0-1 scale)')
            ax2.set_title('Your Performance Factors (Lowest to Highest)')
            
            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax2.text(width + 0.02, bar.get_y() + bar.get_height()/2, 
                        f'{width:.2f}', va='center')
            
            st.pyplot(fig2)
        
        with viz_tab2:
            if feature_importance is not None and prediction_model != "Basic Weighted Formula":
                st.subheader("Model Insights")
                try:
                    # Create a bar chart of feature importance
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Map feature names to display names
                    feature_map = {
                        'Attendance': 'Attendance',
                        'Study_Hours_per_Week': 'Study Hours',
                        'Assignments_Avg': 'Assignments',
                        'Quizzes_Avg': 'Quizzes',
                        'Participation_Score': 'Participation',
                        'Projects_Score': 'Projects',
                        'Midterm_Score': 'Midterm Exam',
                        'Final_Score': 'Final Exam',
                        'Sleep_Hours_per_Night': 'Sleep Quality',
                        'Stress_Level (1-10)': 'Stress Management'
                    }
                    
                    # Create display names for features
                    display_features = [feature_map.get(f, f) for f in feature_importance.keys()]
                    
                    # Sort features by importance
                    sorted_indices = np.argsort(list(feature_importance.values()))
                    sorted_features = [display_features[i] for i in sorted_indices]
                    sorted_importance = [list(feature_importance.values())[i] for i in sorted_indices]
                    
                    # Create horizontal bar chart
                    bars = ax.barh(sorted_features, sorted_importance)
                    
                    # Add value labels
                    for bar in bars:
                        width = bar.get_width()
                        ax.text(width, bar.get_y() + bar.get_height()/2, 
                                f'{width:.3f}', 
                                ha='left', va='center', fontsize=10)
                    
                    ax.set_title('Feature Importance in Prediction')
                    ax.set_xlabel('Importance Score')
                    ax.set_ylabel('Factor')
                    
                    # Adjust layout
                    plt.tight_layout()
                    
                    # Display the chart
                    st.pyplot(fig)
                    
                    # Explain the insights
                    st.write("**How to interpret these insights:**")
                    st.write("The chart shows which factors have the most impact on your predicted score.")
                    st.write("Focus on improving factors with high importance where your performance is low.")
                    
                    # Highlight top 3 most important factors
                    top_3_indices = sorted_indices[-3:]
                    top_3_features = [display_features[i] for i in top_3_indices]
                    
                    st.write("**Top 3 most important factors:**")
                    for i, feature in enumerate(top_3_features):
                        st.write(f"{i+1}. {feature}")
                except Exception as e:
                    st.warning(f"Error displaying model insights: {str(e)}")
                    st.write("Model insights are not available at this time.")
            
            # Store the prediction in session state for future use
            try:
                st.session_state['last_prediction'] = {
                    'predicted_score': predicted_score,
                    'model_used': prediction_model,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            except Exception as e:
                st.warning(f"Error storing prediction: {str(e)}")
        
        # Store prediction in session state for possible future use
        st.session_state.latest_prediction = {
            "student_id": st.session_state.student_id,
            "student_name": st.session_state.student_name,
            "predicted_score": predicted_score,
            "predicted_grade": predicted_grade,
            "factors": all_factors,
            "model_used": prediction_model,
            "ml_accuracy": model_accuracy,
            "input_data": current_data
        }
        
        # Success message
        st.success("Prediction completed! You can adjust your inputs above to see how different factors affect your predicted performance.")
    
    # Footer with navigation
    st.markdown("---")
    col_nav1, col_nav2 = st.columns(2)
    
    with col_nav1:
        if st.button("Log Out"):
            # Clear session state
            for key in ["student_id", "student_name", "latest_prediction", "show_student_input_button"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.switch_page("Home.py")
    
    with col_nav2:
        if st.button("Return to Home"):
            st.switch_page("Home.py") 