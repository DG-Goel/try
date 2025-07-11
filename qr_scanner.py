import os
import logging
import validators
from typing import Optional, List, Dict, Any, Tuple
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
from pyzbar.pyzbar import decode
from pyzbar.pyzbar import ZBarSymbol
import requests
from urllib.parse import urlparse
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QRCodeType:
    """QR Code content types"""
    URL = "url"
    EMAIL = "email"
    PHONE = "phone"
    TEXT = "text"
    WIFI = "wifi"
    VCARD = "vcard"
    LOCATION = "location"
    UNKNOWN = "unknown"


class QRScanner:
    def __init__(self):
        self.supported_formats = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']

    def scan_qr(self, image_path: str, enhance_image: bool = True) -> Dict[str, Any]:
        """
        Scan QR code from image with enhanced error handling and preprocessing

        Args:
            image_path: Path to the image file
            enhance_image: Whether to apply image enhancement for better detection

        Returns:
            Dict containing QR code data and metadata
        """
        try:
            if not os.path.exists(image_path):
                logger.error(f"Image file not found: {image_path}")
                return self._create_error_response("Image file not found")

            # Check file format
            file_ext = os.path.splitext(image_path)[1].lower()
            if file_ext not in self.supported_formats:
                logger.error(f"Unsupported image format: {file_ext}")
                return self._create_error_response(f"Unsupported image format: {file_ext}")

            # Load and process image
            img = Image.open(image_path)

            # Try multiple processing approaches
            approaches = [
                ("original", img),
                ("enhanced", self._enhance_image(img) if enhance_image else img),
                ("grayscale", img.convert('L')),
                ("high_contrast", self._apply_high_contrast(img) if enhance_image else img)
            ]

            for approach_name, processed_img in approaches:
                result = self._decode_qr_codes(processed_img)
                if result["success"]:
                    logger.info(f"QR code detected using {approach_name} approach")
                    result["processing_method"] = approach_name
                    return result

            # If PIL approaches fail, try OpenCV
            if enhance_image:
                cv_result = self._scan_with_opencv(image_path)
                if cv_result["success"]:
                    logger.info("QR code detected using OpenCV approach")
                    return cv_result

            logger.warning(f"No QR code found in image: {image_path}")
            return self._create_error_response("No QR code found in image")

        except Exception as e:
            logger.error(f"Error scanning QR code: {e}")
            return self._create_error_response(f"Error scanning QR code: {str(e)}")

    def _decode_qr_codes(self, img: Image.Image) -> Dict[str, Any]:
        """Decode QR codes from PIL Image"""
        try:
            # Convert PIL image to format compatible with pyzbar
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Decode QR codes
            decoded_objects = decode(img, symbols=[ZBarSymbol.QRCODE])

            if decoded_objects:
                # Process first QR code found
                qr_data = decoded_objects[0].data.decode('utf-8')
                qr_type = self._detect_qr_type(qr_data)

                response = {
                    "success": True,
                    "data": qr_data,
                    "type": qr_type,
                    "count": len(decoded_objects),
                    "all_codes": [obj.data.decode('utf-8') for obj in decoded_objects],
                    "positions": [self._get_qr_position(obj) for obj in decoded_objects]
                }

                # Add type-specific metadata
                if qr_type == QRCodeType.URL:
                    response["url_info"] = self._analyze_url(qr_data)
                elif qr_type == QRCodeType.EMAIL:
                    response["email_info"] = self._parse_email(qr_data)
                elif qr_type == QRCodeType.PHONE:
                    response["phone_info"] = self._parse_phone(qr_data)

                return response

            return {"success": False, "error": "No QR codes found"}

        except Exception as e:
            logger.error(f"Error decoding QR codes: {e}")
            return {"success": False, "error": str(e)}

    def _scan_with_opencv(self, image_path: str) -> Dict[str, Any]:
        """Fallback QR scanning using OpenCV"""
        try:
            # Read image with OpenCV
            img = cv2.imread(image_path)
            if img is None:
                return {"success": False, "error": "Could not read image with OpenCV"}

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Apply various preprocessing techniques
            processed_images = [
                ("original", gray),
                ("gaussian_blur", cv2.GaussianBlur(gray, (5, 5), 0)),
                ("median_blur", cv2.medianBlur(gray, 5)),
                ("bilateral_filter", cv2.bilateralFilter(gray, 9, 75, 75)),
                ("adaptive_threshold",
                 cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2))
            ]

            for method_name, processed_img in processed_images:
                # Convert back to PIL for pyzbar
                pil_img = Image.fromarray(processed_img)
                result = self._decode_qr_codes(pil_img)

                if result["success"]:
                    result["processing_method"] = f"opencv_{method_name}"
                    return result

            return {"success": False, "error": "No QR codes found with OpenCV"}

        except Exception as e:
            logger.error(f"Error with OpenCV scanning: {e}")
            return {"success": False, "error": str(e)}

    def _enhance_image(self, img: Image.Image) -> Image.Image:
        """Apply image enhancement for better QR detection"""
        try:
            # Increase contrast
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.5)

            # Increase sharpness
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(2.0)

            # Apply unsharp mask filter
            img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=100, threshold=1))

            return img
        except Exception as e:
            logger.error(f"Error enhancing image: {e}")
            return img

    def _apply_high_contrast(self, img: Image.Image) -> Image.Image:
        """Apply high contrast processing"""
        try:
            # Convert to grayscale
            gray = img.convert('L')

            # Apply high contrast
            enhancer = ImageEnhance.Contrast(gray)
            return enhancer.enhance(3.0)
        except Exception as e:
            logger.error(f"Error applying high contrast: {e}")
            return img

    def _detect_qr_type(self, data: str) -> str:
        """Detect the type of QR code content"""
        data_lower = data.lower()

        if validators.url(data):
            return QRCodeType.URL
        elif data.startswith('mailto:') or '@' in data:
            return QRCodeType.EMAIL
        elif data.startswith('tel:') or data.startswith('phone:'):
            return QRCodeType.PHONE
        elif data.startswith('wifi:'):
            return QRCodeType.WIFI
        elif data.startswith('begin:vcard'):
            return QRCodeType.VCARD
        elif data.startswith('geo:'):
            return QRCodeType.LOCATION
        else:
            return QRCodeType.TEXT

    def _analyze_url(self, url: str) -> Dict[str, Any]:
        """Analyze URL QR code"""
        try:
            parsed = urlparse(url)
            return {
                "domain": parsed.netloc,
                "scheme": parsed.scheme,
                "path": parsed.path,
                "is_secure": parsed.scheme == 'https',
                "is_valid": validators.url(url)
            }
        except Exception as e:
            logger.error(f"Error analyzing URL: {e}")
            return {"error": str(e)}

    def _parse_email(self, email_data: str) -> Dict[str, Any]:
        """Parse email QR code"""
        try:
            if email_data.startswith('mailto:'):
                email = email_data[7:]  # Remove 'mailto:' prefix
            else:
                email = email_data

            return {
                "email": email,
                "is_valid": validators.email(email)
            }
        except Exception as e:
            logger.error(f"Error parsing email: {e}")
            return {"error": str(e)}

    def _parse_phone(self, phone_data: str) -> Dict[str, Any]:
        """Parse phone QR code"""
        try:
            if phone_data.startswith('tel:'):
                phone = phone_data[4:]  # Remove 'tel:' prefix
            elif phone_data.startswith('phone:'):
                phone = phone_data[6:]  # Remove 'phone:' prefix
            else:
                phone = phone_data

            return {
                "phone": phone,
                "formatted": phone.replace(' ', '').replace('-', '').replace('(', '').replace(')', '')
            }
        except Exception as e:
            logger.error(f"Error parsing phone: {e}")
            return {"error": str(e)}

    def _get_qr_position(self, decoded_obj) -> Dict[str, Any]:
        """Get QR code position information"""
        try:
            points = decoded_obj.polygon
            if points:
                return {
                    "polygon": [(point.x, point.y) for point in points],
                    "bounding_box": {
                        "left": min(point.x for point in points),
                        "top": min(point.y for point in points),
                        "right": max(point.x for point in points),
                        "bottom": max(point.y for point in points)
                    }
                }
            return {}
        except Exception as e:
            logger.error(f"Error getting QR position: {e}")
            return {}

    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            "success": False,
            "error": error_message,
            "data": None,
            "type": None
        }

    def scan_multiple_qr_codes(self, image_path: str) -> List[Dict[str, Any]]:
        """Scan and return all QR codes found in image"""
        try:
            result = self.scan_qr(image_path)
            if result["success"] and result.get("count", 0) > 1:
                return [
                    {
                        "data": code,
                        "type": self._detect_qr_type(code),
                        "position": pos
                    }
                    for code, pos in zip(result["all_codes"], result["positions"])
                ]
            elif result["success"]:
                return [{"data": result["data"], "type": result["type"]}]
            else:
                return []
        except Exception as e:
            logger.error(f"Error scanning multiple QR codes: {e}")
            return []

    def download_and_scan_qr(self, image_url: str) -> Dict[str, Any]:
        """Download image from URL and scan for QR codes"""
        try:
            if not validators.url(image_url):
                return self._create_error_response("Invalid URL provided")

            # Download image
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()

            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                tmp_file.write(response.content)
                tmp_file_path = tmp_file.name

            # Scan QR code
            result = self.scan_qr(tmp_file_path)

            # Clean up
            os.unlink(tmp_file_path)

            return result

        except requests.RequestException as e:
            logger.error(f"Error downloading image: {e}")
            return self._create_error_response(f"Error downloading image: {str(e)}")
        except Exception as e:
            logger.error(f"Error processing downloaded image: {e}")
            return self._create_error_response(f"Error processing image: {str(e)}")


# Global scanner instance
qr_scanner = QRScanner()


def scan_qr(image_path: str) -> str:
    """
    Simple wrapper function for backward compatibility
    Returns just the QR code data as string
    """
    result = qr_scanner.scan_qr(image_path)
    if result["success"]:
        return result["data"]
    else:
        return result.get("error", "No QR code found.")


def scan_qr_detailed(image_path: str) -> Dict[str, Any]:
    """
    Enhanced function that returns detailed QR code information
    """
    return qr_scanner.scan_qr(image_path)


def scan_qr_from_url(image_url: str) -> Dict[str, Any]:
    """
    Scan QR code from image URL
    """
    return qr_scanner.download_and_scan_qr(image_url)


def get_qr_info(qr_data: str) -> Dict[str, Any]:
    """
    Get information about QR code content without scanning
    """
    return {
        "data": qr_data,
        "type": qr_scanner._detect_qr_type(qr_data),
        "length": len(qr_data)
    }
