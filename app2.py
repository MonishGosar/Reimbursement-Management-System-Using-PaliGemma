import google.generativeai as genai
import sqlite3
import streamlit as st
from PIL import Image
import pandas as pd
import os
import re
from datetime import datetime
import io
import cv2

st.set_page_config(layout="wide")

# Google Gemini Configuration
GOOGLE_API_KEY = 'AIzaSyAPyak054If8ttTbkelk-5Ht0PkOxAzSL4'  
genai.configure(api_key=GOOGLE_API_KEY)

# Utility Functions
def parse_date(date_str):
    """Parse date string in various formats."""
    try:
        formats = [
            '%d-%m-%Y',    # 26-07-2024
            '%Y-%m-%d',    # 2024-07-26
            '%m/%d/%Y',    # 07/26/2024
            '%d/%m/%Y',    # 26/07/2024
            '%Y/%m/%d',    # 2024/07/26
            '%b %d, %Y',   # Jul 26, 2024
            '%d %b %Y',    # 26 Jul 2024
            '%Y-%m-%d %H:%M:%S'  # 2024-07-26 12:34:56
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt).strftime('%Y-%m-%d')
            except ValueError:
                continue
        
        return datetime.now().strftime('%Y-%m-%d')
    except Exception:
        return datetime.now().strftime('%Y-%m-%d')

def clean_amount(amount_str):
    """Clean and parse amount string."""
    try:
        cleaned = re.sub(r'[^\d.,]', '', str(amount_str))
        if ',' in cleaned and '.' not in cleaned:
            cleaned = cleaned.replace(',', '.')
        cleaned = cleaned.replace(',', '')
        return float(cleaned)
    except (ValueError, TypeError):
        return 0.0

# Database Functions
def init_database():
    """Initialize SQLite database with required tables."""
    conn = sqlite3.connect('reimbursements.db')
    c = conn.cursor()
    
    try:
        # Check existing tables
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='receipts'")
        table_exists = c.fetchone() is not None
        
        if not table_exists:
            c.execute('''CREATE TABLE IF NOT EXISTS receipts (
                         id INTEGER PRIMARY KEY AUTOINCREMENT,
                         employee_id TEXT,
                         category TEXT,
                         total REAL,
                         currency TEXT,
                         date TEXT,
                         status TEXT DEFAULT 'Pending',
                         receipt_image BLOB,
                         submission_date TEXT)''')
        else:
            # Check if submission_date column exists
            c.execute("PRAGMA table_info(receipts)")
            columns = [column[1] for column in c.fetchall()]
            if 'submission_date' not in columns:
                c.execute('ALTER TABLE receipts ADD COLUMN submission_date TEXT')
                c.execute("UPDATE receipts SET submission_date = ?", 
                         (datetime.now().strftime('%Y-%m-%d'),))
        
        conn.commit()
    except Exception as e:
        st.error(f"Database initialization error: {str(e)}")
    finally:
        conn.close()

def save_receipt(employee_id, category, total, currency, date, image_path):
    """Save receipt data to database."""
    try:
        conn = sqlite3.connect('reimbursements.db')
        c = conn.cursor()
        with open(image_path, 'rb') as f:
            image_blob = f.read()
        submission_date = datetime.now().strftime('%Y-%m-%d')
        c.execute('''INSERT INTO receipts 
                    (employee_id, category, total, currency, date, receipt_image, submission_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?)''',
                    (employee_id, category, total, currency, date, image_blob, submission_date))
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Error saving receipt: {str(e)}")
        return False
    finally:
        if conn:
            conn.close()

def load_receipts_data():
    """Load receipts data from database."""
    conn = sqlite3.connect('reimbursements.db')
    try:
        receipts_df = pd.read_sql_query("""
            SELECT id, employee_id, category, total, currency, date, status, 
                   COALESCE(submission_date, date) as submission_date 
            FROM receipts
        """, conn)
    except sqlite3.OperationalError:
        # Fallback query without submission_date
        receipts_df = pd.read_sql_query("""
            SELECT id, employee_id, category, total, currency, date, status
            FROM receipts
        """, conn)
        receipts_df['submission_date'] = receipts_df['date']
    finally:
        conn.close()
    return receipts_df

# Image Processing Functions
def prep_image(image_path):
    """Prepare image for Google Gemini processing."""
    try:
        sample_file = genai.upload_file(path=image_path, display_name="Receipt")
        return sample_file
    except Exception as e:
        st.error(f"Error preparing image: {str(e)}")
        return None

def extract_text_from_image(image_path, prompt):
    """Extract text from image using Google Gemini API."""
    try:
        model = genai.GenerativeModel(model_name="gemini-1.5-pro")
        response = model.generate_content([image_path, prompt])
        return response.text if response else None
    except Exception as e:
        st.error(f"Error extracting text: {str(e)}")
        return None

def parse_extracted_text(text):
    """Parse the extracted text into structured data."""
    data = {
        'category': 'Unknown',
        'total': 0.0,
        'currency': 'INR',
        'date': datetime.now().strftime('%Y-%m-%d')
    }
    
    if text:
        # Extract receipt type/category
        category_match = re.search(r'Receipt Type:?\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
        if category_match:
            category = category_match.group(1).strip()
            data['category'] = category if category else 'Unknown'
        
        # Extract total amount
        total_match = re.search(r'(?:Grand )?Total:?\s*(?:₹|Rs\.?)?\s*([\d,]+\.?\d*)', text, re.IGNORECASE)
        if total_match:
            amount_str = total_match.group(1).replace(',', '')
            try:
                data['total'] = float(amount_str)
            except ValueError:
                data['total'] = 0.0
        
        # Set currency
        if '₹' in text or 'Rs.' in text:
            data['currency'] = 'INR'
        
        # Extract date
        date_match = re.search(r'Date of Receipt:?\s*(\d{2}[-/]\w{3}[-/]\d{4})', text, re.IGNORECASE)
        if date_match:
            try:
                date_str = date_match.group(1)
                parsed_date = datetime.strptime(date_str, '%d-%b-%Y')
                data['date'] = parsed_date.strftime('%Y-%m-%d')
            except ValueError:
                data['date'] = datetime.now().strftime('%Y-%m-%d')
    
    return data

def capture_image():
    """Capture image from webcam."""
    img_file_buffer = st.camera_input(
        "Take a picture of your receipt",
        key="camera_input",
        help="Center the receipt in the frame and ensure good lighting"
    )
    
    if img_file_buffer is not None:
        # Save the captured image temporarily
        image = Image.open(img_file_buffer)
        temp_file = f"temp_capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        image.save(temp_file)
        return temp_file, img_file_buffer
    return None, None

def safe_process_dataframe(df):
    """Safely process dataframe for analytics."""
    try:
        if df.empty:
            return pd.DataFrame()

        processed_df = df.copy()
        processed_df['date'] = processed_df['date'].apply(parse_date)
        processed_df['total'] = processed_df['total'].apply(clean_amount)
        processed_df['month'] = pd.to_datetime(processed_df['date']).dt.strftime('%Y-%m')
        
        return processed_df
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return df

# Initialize the database
init_database()

# Streamlit UI
st.title("Reimbursement Management System")

# Sidebar for employee section
st.sidebar.header("Employee Section")
employee_id = st.sidebar.text_input("Employee ID")

# Add option to choose between upload and camera
receipt_input_method = st.sidebar.radio(
    "Choose how to submit receipt",
    ("Upload File", "Use Camera")
)

if receipt_input_method == "Upload File":
    uploaded_file = st.sidebar.file_uploader("Upload your receipt", type=["jpg", "jpeg", "png", "pdf"])
    file_path = None
    if uploaded_file:
        file_path = f"temp_upload_{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
else:
    uploaded_file = None
    file_path, camera_file = capture_image()

# Main content area
tab1, tab2 = st.tabs(["Submit Receipt", "Manager Dashboard"])

# Submit Receipt Tab
with tab1:
    if employee_id and (uploaded_file or (receipt_input_method == "Use Camera" and file_path)):
        try:
            # Display image
            if receipt_input_method == "Upload File":
                image = Image.open(uploaded_file)
            else:
                image = Image.open(file_path)
            st.image(image, caption="Receipt Image", use_column_width=True)

            # Process image
            sample_file = prep_image(file_path)
            if sample_file:
                prompt = """
                Please extract the following information from this receipt in exactly this format:
                Receipt Type: [type]
                Grand Total: [amount with currency symbol]
                Date of Receipt: [date in DD-MMM-YYYY format]

                Important:
                - Keep the exact labels as shown
                - Include the currency symbol (₹) if present
                - Use DD-MMM-YYYY format for date (e.g., 06-Jun-2024)
                - Extract complete values without modifications
                """
                extracted_text = extract_text_from_image(sample_file, prompt)

                if extracted_text:
                    st.subheader("Extracted Information")
                    st.text(extracted_text)
                    
                    # Parse and display extracted data
                    extracted_data = parse_extracted_text(extracted_text)
                    st.subheader("Parsed Data")
                    for key, value in extracted_data.items():
                        st.write(f"{key}: {value}")
                    
                    # Allow manual corrections
                    st.subheader("Edit Extracted Data")
                    corrected_data = {}
                    corrected_data['category'] = st.text_input("Category", value=extracted_data['category'])
                    corrected_data['total'] = st.number_input("Total Amount", value=float(extracted_data['total']))
                    corrected_data['currency'] = st.selectbox("Currency", ['INR', 'USD', 'EUR'], index=0)
                    corrected_data['date'] = st.date_input("Date", value=datetime.strptime(extracted_data['date'], '%Y-%m-%d')).strftime('%Y-%m-%d')
                    
                    # Save button
                    if st.button("Submit Receipt"):
                        if save_receipt(
                            employee_id,
                            corrected_data['category'],
                            corrected_data['total'],
                            corrected_data['currency'],
                            corrected_data['date'],
                            file_path
                        ):
                            st.success("Receipt submitted successfully!")
                        else:
                            st.error("Failed to save receipt.")

        except Exception as e:
            st.error(f"Error processing receipt: {str(e)}")
        finally:
            # Cleanup temporary files
            if file_path and os.path.exists(file_path):
                os.remove(file_path)

# Manager Dashboard Tab
with tab2:
    st.header("Manager Dashboard")
    
    # Load and process data
    receipts_df = load_receipts_data()
    processed_df = safe_process_dataframe(receipts_df)

    if not processed_df.empty:
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            categories = ['All'] + sorted(list(processed_df['category'].unique()))
            category_filter = st.selectbox("Filter by Category", categories)
        with col2:
            statuses = ['All', 'Pending', 'Approved', 'Rejected']
            status_filter = st.selectbox("Filter by Status", statuses)
        with col3:
            employees = ['All'] + sorted(list(processed_df['employee_id'].unique()))
            employee_filter = st.selectbox("Filter by Employee", employees)

        # Apply filters
        filtered_df = processed_df.copy()
        if category_filter != "All":
            filtered_df = filtered_df[filtered_df['category'] == category_filter]
        if status_filter != "All":
            filtered_df = filtered_df[filtered_df['status'] == status_filter]
        if employee_filter != "All":
            filtered_df = filtered_df[filtered_df['employee_id'] == employee_filter]

        # Display filtered data
        st.subheader("Receipt Records")
        display_df = filtered_df.drop('receipt_image', axis=1, errors='ignore')
        st.dataframe(display_df)

        # Approval/Rejection Section
        st.subheader("Approve or Reject Receipts")
        selected_receipts = st.multiselect(
            "Select Receipts to Process",
            filtered_df['id'].tolist(),
            format_func=lambda x: f"Receipt {x}"
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Approve Selected", type="primary"):
                conn = sqlite3.connect('reimbursements.db')
                try:
                    for receipt_id in selected_receipts:
                        conn.execute(
                            "UPDATE receipts SET status = 'Approved' WHERE id = ?",
                            (receipt_id,)
                        )
                    conn.commit()
                    st.success("Selected receipts approved!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error approving receipts: {str(e)}")
                finally:
                    conn.close()

        with col2:
            if st.button("Reject Selected", type="secondary"):
                conn = sqlite3.connect('reimbursements.db')
                try:
                    for receipt_id in selected_receipts:
                        conn.execute("UPDATE receipts SET status = 'Rejected' WHERE id = ?",
                            (receipt_id,)
                        )
                    conn.commit()
                    st.success("Selected receipts rejected!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error rejecting receipts: {str(e)}")
                finally:
                    conn.close()

        # Analytics Section
        if not filtered_df.empty:
            st.header("Analytics")
            
            # Time series of expenses
            st.subheader("Monthly Expense Trends")
            try:
                monthly_spending = filtered_df.groupby('month')['total'].sum().sort_index()
                if not monthly_spending.empty:
                    st.line_chart(monthly_spending)
                else:
                    st.info("No expense data available for the selected filters.")
            except Exception as e:
                st.error(f"Error creating expense trends chart: {str(e)}")

            # Category breakdown and Status distribution
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Spending by Category")
                try:
                    category_spending = filtered_df.groupby('category')['total'].sum()
                    if not category_spending.empty:
                        st.bar_chart(category_spending)
                    else:
                        st.info("No category data available.")
                except Exception as e:
                    st.error(f"Error creating category chart: {str(e)}")

            with col2:
                st.subheader("Status Distribution")
                try:
                    status_counts = filtered_df['status'].value_counts()
                    if not status_counts.empty:
                        st.bar_chart(status_counts)
                    else:
                        st.info("No status data available.")
                except Exception as e:
                    st.error(f"Error creating status chart: {str(e)}")

            # Summary statistics
            st.subheader("Summary Statistics")
            try:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Receipts", len(filtered_df))
                with col2:
                    total_amount = filtered_df['total'].sum()
                    st.metric("Total Amount", f"{filtered_df['currency'].iloc[0]} {total_amount:,.2f}")
                with col3:
                    avg_amount = filtered_df['total'].mean()
                    st.metric("Average Amount", f"{filtered_df['currency'].iloc[0]} {avg_amount:,.2f}")
                with col4:
                    pending_count = len(filtered_df[filtered_df['status'] == 'Pending'])
                    st.metric("Pending Approvals", pending_count)
            except Exception as e:
                st.error(f"Error calculating summary statistics: {str(e)}")

            # Advanced Analytics
            st.subheader("Advanced Analytics")
            
            # Month-over-month growth
            try:
                monthly_totals = filtered_df.groupby('month')['total'].sum()
                mom_growth = monthly_totals.pct_change() * 100
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Month-over-Month Growth")
                    if not mom_growth.empty:
                        st.line_chart(mom_growth)
                    else:
                        st.info("Insufficient data for growth calculation.")
                
                with col2:
                    st.subheader("Top Spenders")
                    top_spenders = filtered_df.groupby('employee_id')['total'].sum().sort_values(ascending=False).head(5)
                    st.bar_chart(top_spenders)
            except Exception as e:
                st.error(f"Error calculating advanced analytics: {str(e)}")

            # Export functionality
            st.subheader("Export Data")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Export to CSV"):
                    try:
                        # Prepare data for export
                        export_df = filtered_df.drop('receipt_image', axis=1, errors='ignore')
                        csv = export_df.to_csv(index=False)
                        
                        # Create download button
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name=f"reimbursements_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )
                    except Exception as e:
                        st.error(f"Error exporting data: {str(e)}")
            
            with col2:
                if st.button("Export Analytics Report"):
                    try:
                        # Create a comprehensive report
                        report = f"""Reimbursement Analytics Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Summary Statistics:
- Total Receipts: {len(filtered_df)}
- Total Amount: {filtered_df['currency'].iloc[0]} {total_amount:,.2f}
- Average Amount: {filtered_df['currency'].iloc[0]} {avg_amount:,.2f}
- Pending Approvals: {pending_count}

Category Breakdown:
{category_spending.to_string()}

Status Distribution:
{status_counts.to_string()}

Monthly Spending:
{monthly_spending.to_string()}
"""
                        # Create download button for report
                        st.download_button(
                            label="Download Report",
                            data=report,
                            file_name=f"reimbursement_report_{datetime.now().strftime('%Y%m%d')}.txt",
                            mime="text/plain"
                        )
                    except Exception as e:
                        st.error(f"Error exporting analytics report: {str(e)}")

    else:
        st.info("No receipt data available yet. Upload some receipts to see analytics.")

    # Add refresh button
    if st.button("Refresh Dashboard"):
        st.rerun()

# Add footer with system information
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <small>Reimbursement Management System • Last updated: {}</small>
</div>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)

# Session state management for notifications
if 'notifications' not in st.session_state:
    st.session_state.notifications = []

# Display notifications if any
for notification in st.session_state.notifications:
    st.toast(notification)
st.session_state.notifications = []  # Clear notifications after displaying

# Error handling for the entire app
try:
    # Check database connection
    conn = sqlite3.connect('reimbursements.db')
    conn.close()
except Exception as e:
    st.error(f"""
    Database connection error. Please check if:
    1. The database file exists
    2. You have write permissions
    3. There's enough disk space
    
    Error details: {str(e)}
    """)

# Add keyboard shortcuts
st.markdown("""
<script>
document.addEventListener('keydown', function(e) {
    if (e.ctrlKey && e.key === 'r') {  // Ctrl+R to refresh
        window.location.reload();
    }
});
</script>
""", unsafe_allow_html=True)