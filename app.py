# C:/Projects/careerqr/app.py
import streamlit as st
import os
import tempfile
import requests
import socket
import qrcode
import io
import re  # Import the regular expression module

# Your existing custom modules
from qr_scanner import scan_qr
from azure_resume_parser import extract_resume_data_full
from azure_ai_advisor import get_career_advice
# The import from azure_speaker is already correct
from azure_speaker import get_speech_audio_data

# --- Constants ---
PAGE_TITLE = "CareerQR"
PAGE_ICON = "🧠"
APP_TITLE = "📄 CareerQR – Scan Your Future"
APP_SUBHEADER = "Upload a QR code with a resume link, or upload a resume PDF directly to get AI-powered career advice."
QR_FILE_TYPES = ["png", "jpg", "jpeg"]
RESUME_FILE_TYPES = ["pdf"]


# --- Helper Functions ---

def clean_text_for_speech(text: str) -> str:
    """
    Removes Markdown and other non-verbal characters from text to improve speech synthesis.
    """
    # Remove Markdown headings, bold, italics, etc.
    text = re.sub(r'#{1,6} ', '', text)  # Headings like #, ##
    text = re.sub(r'(\*\*|__)(.*?)(\*\*|__)', r'\2', text)  # **Bold** or __Bold__
    text = re.sub(r'(\*|_)(.*?)(\*|_)', r'\2', text)  # *Italic* or _Italic_

    # Remove table formatting characters and code backticks
    text = text.replace('|', ' ').replace('---', ' ').replace('`', '')

    # Collapse multiple newlines and spaces into a single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ... (keep get_local_ip, display_network_qr_code, analyze_resume, etc. as they are) ...
def get_local_ip():
    """
    Finds the local IP address of the machine.
    Returns '127.0.0.1' if the IP cannot be determined.
    """
    # Using a 'with' statement ensures the socket is always closed properly.
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            # This doesn't need to be a reachable address, just a valid target.
            s.connect(('10.255.255.255', 1))
            IP = s.getsockname()[0]
        except Exception:
            IP = '127.0.0.1'
    return IP


def display_network_qr_code():
    """
    Generates and displays a QR code in the sidebar for accessing the app on the local network.
    """
    try:
        # A more robust way to get the port, with a fallback.
        from streamlit.web.server.server import Server
        port = Server.get_current()._port
    except (ImportError, AttributeError):
        port = 8501

    local_ip = get_local_ip()

    if local_ip != '127.0.0.1':
        network_url = f"http://{local_ip}:{port}"
        st.sidebar.header("📱 Access on Your Phone")
        st.sidebar.info(
            "Scan this QR code with your phone to open this app on your mobile browser (requires same Wi-Fi).")

        qr_img = qrcode.make(network_url)
        img_buffer = io.BytesIO()
        qr_img.save(img_buffer, format="PNG")

        st.sidebar.image(img_buffer, use_column_width=True)
        st.sidebar.code(network_url)
    else:
        st.sidebar.warning(
            "Could not determine local network IP. To enable phone access, run with `--server.address=0.0.0.0`.")


def analyze_resume(resume_path: str):
    """Analyzes a resume file and stores results in session state."""
    try:
        with st.spinner("⏳ Analyzing your resume with Azure Form Recognizer..."):
            resume_data = extract_resume_data_full(resume_path)
            if not resume_data:
                st.error("Could not extract data from resume. Please check the file or try another.")
                return

        st.session_state.resume_data = resume_data

        with st.spinner("🧠 Generating career advice with Azure OpenAI..."):
            advice = get_career_advice(resume_data)
            if not advice or "error" in advice.lower():
                st.error("Could not generate career advice. Please try again.")
                return

        st.session_state.advice = advice
    except Exception as e:
        st.error(f"An error occurred during analysis: {str(e)}")


def handle_file_upload(uploaded_file, handler_func):
    """Generic handler for file uploads to avoid code repetition."""
    # This approach ensures the temporary file is always cleaned up, even if analysis fails.
    file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            file_path = tmp_file.name
            tmp_file.write(uploaded_file.getvalue())

        handler_func(file_path)
    finally:
        if file_path and os.path.exists(file_path):
            os.unlink(file_path)


def handle_qr_code_upload(qr_file):
    """Processes an uploaded QR code image."""

    def process_qr(image_path):
        qr_content = scan_qr(image_path)
        if not qr_content or "error" in qr_content.lower():
            st.error("Could not decode QR code. Please try another image.")
            return

        st.info(f"🔗 Decoded QR content: {qr_content}")

        if qr_content.startswith(('http://', 'https://')):
            with st.spinner("Downloading resume from URL..."):
                pdf_path = None
                try:
                    response = requests.get(qr_content, stream=True, timeout=10)
                    response.raise_for_status()

                    # Create a temporary file for the downloaded PDF
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_pdf:
                        pdf_path = tmp_pdf.name
                        for chunk in response.iter_content(chunk_size=8192):
                            tmp_pdf.write(chunk)

                    analyze_resume(pdf_path)

                except requests.exceptions.RequestException as e:
                    st.error(f"Failed to download file from URL: {str(e)}")
                finally:
                    # Ensure the downloaded file is always cleaned up
                    if pdf_path and os.path.exists(pdf_path):
                        os.unlink(pdf_path)
        else:
            st.warning("The QR code did not contain a valid URL. Please upload a resume PDF directly.")

    handle_file_upload(qr_file, process_qr)


def handle_resume_upload(resume_file):
    """Processes a directly uploaded resume PDF."""
    handle_file_upload(resume_file, analyze_resume)


# --- Streamlit App UI ---

def initialize_state():
    """Initializes session state variables if they don't exist."""
    if "advice" not in st.session_state:
        st.session_state.advice = None
    if "resume_data" not in st.session_state:
        st.session_state.resume_data = None


def reset_app():
    """Clears session state to reset the application."""
    keys_to_clear = ["advice", "resume_data"]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()


def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")

    display_network_qr_code()
    st.title(APP_TITLE)
    st.markdown(APP_SUBHEADER)

    initialize_state()

    if st.session_state.advice:
        # --- Results View ---
        st.header("Analysis Complete!")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Your AI Career Advice")
            st.markdown(st.session_state.advice, unsafe_allow_html=True)

            # --- MODIFIED BUTTON LOGIC ---
            if st.button("🔊 Speak Advice"):
                with st.spinner("Generating audio..."):
                    # 1. Clean the Markdown text for better speech synthesis
                    advice_text = st.session_state.advice
                    clean_advice = clean_text_for_speech(advice_text)

                    # 2. Get the audio data from Azure using the cleaned text
                    audio_data = get_speech_audio_data(clean_advice)

                    # 3. If successful, display the audio player
                    if audio_data:
                        st.audio(audio_data, format="audio/wav")
                    else:
                        st.error(
                            "Sorry, could not generate audio at this time. Please check the terminal for error details.")
            # --- END OF MODIFICATION ---

        with col2:
            st.subheader("Extracted Resume Data")
            st.json(st.session_state.resume_data)

        if st.button("✨ Start Over"):
            reset_app()
    else:
        # --- Upload View ---
        st.header("Step 1: Provide Your Resume")
        col1, col2 = st.columns(2)

        with col1:
            st.info("Option A: Scan a QR Code")
            qr_file = st.file_uploader("Upload QR Code Image", type=QR_FILE_TYPES, key="qr_uploader")
            if qr_file:
                handle_qr_code_upload(qr_file)

        with col2:
            st.info("Option B: Upload a PDF")
            resume_file = st.file_uploader("Upload Resume (PDF)", type=RESUME_FILE_TYPES, key="resume_uploader")
            if resume_file:
                handle_resume_upload(resume_file)


if __name__ == "__main__":
    main()