# qr_scanner.py
"""
QR Scanner Module for CareerQR
-------------------------------
This module provides QR code scanning functionality using OpenCV's QRCodeDetector.
It was rewritten to avoid the pyzbar/zbar dependency, which causes ImportError on
Azure App Services (since libzbar is not available there).

Functions
---------
- load_image(path): Safely loads an image file.
- preprocess_image(img): Prepares an image for QR detection (optional grayscale).
- scan_qr(image_path): Detects and decodes QR codes from an image file.
- scan_qr_bytes(image_bytes): Alternative QR decoding directly from bytes.
"""

import cv2
import numpy as np
import os
import logging

# Configure logging (important for debugging in Azure logs)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# -----------------------------
# Utility Functions
# -----------------------------

def load_image(image_path: str):
    """
    Safely loads an image from disk using OpenCV.

    Parameters
    ----------
    image_path : str
        Path to the image file.

    Returns
    -------
    img : np.ndarray or None
        The loaded image as a NumPy array, or None if loading failed.
    """
    if not os.path.exists(image_path):
        logging.error(f"Image file does not exist: {image_path}")
        return None

    img = cv2.imread(image_path)
    if img is None:
        logging.error(f"Failed to load image with OpenCV: {image_path}")
    else:
        logging.info(f"Successfully loaded image: {image_path}")
    return img


def preprocess_image(img):
    """
    Optionally preprocesses the image for better QR detection.

    Currently:
    - Converts to grayscale if needed.

    Parameters
    ----------
    img : np.ndarray
        Input image.

    Returns
    -------
    processed_img : np.ndarray
        Grayscale or original image.
    """
    if img is None:
        return None
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        logging.debug("Image converted to grayscale for QR detection.")
        return gray
    except Exception as e:
        logging.warning(f"Grayscale conversion failed, using original image: {e}")
        return img


# -----------------------------
# Main QR Scanning Functions
# -----------------------------

def scan_qr(image_path: str) -> str:
    """
    Detects and decodes a QR code from a file path.

    Parameters
    ----------
    image_path : str
        Path to the QR code image.

    Returns
    -------
    str
        Decoded QR content, or an error message if detection fails.
    """
    try:
        img = load_image(image_path)
        if img is None:
            return "Error: Could not read image file."

        # Preprocess the image (grayscale conversion)
        processed_img = preprocess_image(img)

        # Initialize OpenCV's QRCodeDetector
        detector = cv2.QRCodeDetector()

        # Detect and decode
        data, points, _ = detector.detectAndDecode(processed_img)

        if points is not None and data:
            logging.info(f"QR code detected and decoded: {data[:50]}...")  # limit log length
            return data
        else:
            logging.warning("No QR code found in the image.")
            return "Error: No QR code found in the image."
    except Exception as e:
        logging.exception("Exception during QR decoding.")
        return f"Error while decoding QR: {str(e)}"


def scan_qr_bytes(image_bytes: bytes) -> str:
    """
    Alternative QR scanner that works directly with raw image bytes.

    Parameters
    ----------
    image_bytes : bytes
        Image file content in memory.

    Returns
    -------
    str
        Decoded QR content, or an error message.
    """
    try:
        # Convert bytes to NumPy array
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            logging.error("Failed to decode image bytes.")
            return "Error: Could not decode image bytes."

        processed_img = preprocess_image(img)
        detector = cv2.QRCodeDetector()
        data, points, _ = detector.detectAndDecode(processed_img)

        if points is not None and data:
            logging.info(f"QR code decoded from bytes: {data[:50]}...")
            return data
        else:
            return "Error: No QR code found in the provided bytes."
    except Exception as e:
        logging.exception("Exception during QR decoding from bytes.")
        return f"Error while decoding QR from bytes: {str(e)}"


# -----------------------------
# Test Harness
# -----------------------------

if __name__ == "__main__":
    """
    Basic test harness for local debugging.
    Run this file directly to test QR scanning functionality.
    """
    sample_path = "sample_qr.png"  # Change to your test QR image path
    if os.path.exists(sample_path):
        result = scan_qr(sample_path)
        print("Decoded QR content:", result)
    else:
        print("No sample QR image found for testing.")
