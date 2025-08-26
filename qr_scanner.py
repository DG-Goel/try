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
        """
        try:
            if not os.path.exists(image_path):
                logger.error(f"Image file not found: {image_path}")
                return self._create_error_response("Image file not found")

            file_ext = os.path.splitext(image_path)[1].lower()
            if file_ext not in self.supported_formats:
                logger.error(f"Unsupported image format: {file_ext}")
                return self._create_error_response(f"Unsupported image format: {file_ext}")

            try:
                img = Image.open(image_path)
            except Exception as e:
                logger.error(f"Error opening image: {e}")
                return self._create_error_response(f"Error opening image: {str(e)}")

            approaches = [
                ("original", img),
                ("enhanced", self._enhance_image(img) if enhance_image else img),
                ("grayscale", img.convert('L')),
                ("high_contrast", self._apply_high_contrast(img) if enhance_image else img)
            ]

            for approach_name, processed_img in approaches:
                try:
                    result = self._decode_qr_codes(processed_img)
                    if result["success"]:
                        logger.info(f"QR code detected using {approach_name} approach")
                        result["processing_method"] = approach_name
                        return result
                except Exception as e:
                    logger.error(f"Error decoding with {approach_name} approach: {e}")

            if enhance_image:
                try:
                    cv_result = self._scan_with_opencv(image_path)
                    if cv_result["success"]:
                        logger.info("QR code detected using OpenCV approach")
                        return cv_result
                except Exception as e:
                    logger.error(f"Error scanning with OpenCV: {e}")

            logger.warning(f"No QR code found in image: {image_path}")
            return self._create_error_response("No QR code found in image")

        except Exception as e:
            logger.error(f"Unexpected error scanning QR code: {e}")
            return self._create_error_response(f"Error scanning QR code: {str(e)}")

    def _decode_qr_codes(self, img: Image.Image) -> Dict[str, Any]:
        """Decode QR codes from PIL Image"""
        try:
            if img.mode != 'RGB':
                try:
                    img = img.convert('RGB')
                except Exception as e:
                    logger.error(f"Error converting image to RGB: {e}")
                    return {"success": False, "error": "Failed to convert image"}

            try:
                decoded_objects = decode(img, symbols=[ZBarSymbol.QRCODE])
            except Exception as e:
                logger.error(f"Error during pyzbar decode: {e}")
                return {"success": False, "error": str(e)}

            if decoded_objects:
                try:
                    qr_data = decoded_objects[0].data.decode('utf-8')
                except Exception as e:
                    logger.error(f"Error decoding QR data: {e}")
                    return {"success": False, "error": "Failed to decode QR content"}

                qr_type = self._detect_qr_type(qr_data)

                response = {
                    "success": True,
                    "data": qr_data,
                    "type": qr_type,
                    "count": len(decoded_objects),
                    "all_codes": [],
                    "positions": []
                }

                try:
                    response["all_codes"] = [obj.data.decode('utf-8') for obj in decoded_objects]
                except Exception as e:
                    logger.warning(f"Error extracting all codes: {e}")

                try:
                    response["positions"] = [self._get_qr_position(obj) for obj in decoded_objects]
                except Exception as e:
                    logger.warning(f"Error extracting QR positions: {e}")

                try:
                    if qr_type == QRCodeType.URL:
                        response["url_info"] = self._analyze_url(qr_data)
                    elif qr_type == QRCodeType.EMAIL:
                        response["email_info"] = self._parse_email(qr_data)
                    elif qr_type == QRCodeType.PHONE:
                        response["phone_info"] = self._parse_phone(qr_data)
                except Exception as e:
                    logger.warning(f"Error parsing type-specific info: {e}")

                return response

            return {"success": False, "error": "No QR codes found"}

        except Exception as e:
            logger.error(f"Error decoding QR codes: {e}")
            return {"success": False, "error": str(e)}

    def _scan_with_opencv(self, image_path: str) -> Dict[str, Any]:
        """Fallback QR scanning using OpenCV"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return {"success": False, "error": "Could not read image with OpenCV"}

            try:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            except Exception as e:
                logger.error(f"Error converting to grayscale: {e}")
                return {"success": False, "error": "Failed to preprocess image"}

            processed_images = []
            try:
                processed_images = [
                    ("original", gray),
                    ("gaussian_blur", cv2.GaussianBlur(gray, (5, 5), 0)),
                    ("median_blur", cv2.medianBlur(gray, 5)),
                    ("bilateral_filter", cv2.bilateralFilter(gray, 9, 75, 75)),
                    ("adaptive_threshold",
                     cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 11, 2))
                ]
            except Exception as e:
                logger.error(f"Error applying OpenCV preprocessing: {e}")

            for method_name, processed_img in processed_images:
                try:
                    pil_img = Image.fromarray(processed_img)
                    result = self._decode_qr_codes(pil_img)
                    if result["success"]:
                        result["processing_method"] = f"opencv_{method_name}"
                        return result
                except Exception as e:
                    logger.warning(f"OpenCV method {method_name} failed: {e}")

            return {"success": False, "error": "No QR codes found with OpenCV"}

        except Exception as e:
            logger.error(f"Error with OpenCV scanning: {e}")
            return {"success": False, "error": str(e)}

    def _enhance_image(self, img: Image.Image) -> Image.Image:
        """Apply image enhancement for better QR detection"""
        try:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.5)
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(2.0)
            img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=100, threshold=1))
            return img
        except Exception as e:
            logger.error(f"Error enhancing image: {e}")
            return img

    def _apply_high_contrast(self, img: Image.Image) -> Image.Image:
        """Apply high contrast processing"""
        try:
            gray = img.convert('L')
            enhancer = ImageEnhance.Contrast(gray)
            return enhancer.enhance(3.0)
        except Exception as e:
            logger.error(f"Error applying high contrast: {e}")
            return img

    def _detect_qr_type(self, data: str) -> str:
        """Detect the type of QR code content"""
        try:
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
        except Exception as e:
            logger.error(f"Error detecting QR type: {e}")
            return QRCodeType.UNKNOWN

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
                email = email_data[7:]
            else:
                email = email_data
            return {"email": email, "is_valid": validators.email(email)}
        except Exception as e:
            logger.error(f"Error parsing email: {e}")
            return {"error": str(e)}

    def _parse_phone(self, phone_data: str) -> Dict[str, Any]:
        """Parse phone QR code"""
        try:
            if phone_data.startswith('tel:'):
                phone = phone_data[4:]
            elif phone_data.startswith('phone:'):
                phone = phone_data[6:]
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
        return {"success": False, "error": error_message, "data": None, "type": None}

    def scan_multiple_qr_codes(self, image_path: str) -> List[Dict[str, Any]]:
        try:
            result = self.scan_qr(image_path)
            if result["success"] and result.get("count", 0) > 1:
                try:
                    return [
                        {"data": code, "type": self._detect_qr_type(code), "position": pos}
                        for code, pos in zip(result["all_codes"], result["positions"])
                    ]
                except Exception as e:
                    logger.error(f"Error processing multiple QR codes: {e}")
                    return []
            elif result["success"]:
                return [{"data": result["data"], "type": result["type"]}]
            else:
                return []
        except Exception as e:
            logger.error(f"Error scanning multiple QR codes: {e}")
            return []

    def download_and_scan_qr(self, image_url: str) -> Dict[str, Any]:
        try:
            if not validators.url(image_url):
                return self._create_error_response("Invalid URL provided")

            try:
                response = requests.get(image_url, timeout=10)
                response.raise_for_status()
            except requests.RequestException as e:
                logger.error(f"Error downloading image: {e}")
                return self._create_error_response(f"Error downloading image: {str(e)}")

            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                    tmp_file.write(response.content)
                    tmp_file_path = tmp_file.name
            except Exception as e:
                logger.error(f"Error saving temporary image: {e}")
                return self._create_error_response("Failed to save temporary file")

            try:
                result = self.scan_qr(tmp_file_path)
            except Exception as e:
                logger.error(f"Error scanning downloaded image: {e}")
                result = self._create_error_response("Error scanning downloaded image")

            try:
                os.unlink(tmp_file_path)
            except Exception as e:
                logger.warning(f"Could not delete temporary file: {e}")

            return result

        except Exception as e:
            logger.error(f"Error processing downloaded image: {e}")
            return self._create_error_response(f"Error processing image: {str(e)}")


qr_scanner = QRScanner()


def scan_qr(image_path: str) -> str:
    try:
        result = qr_scanner.scan_qr(image_path)
        if result["success"]:
            return result["data"]
        else:
            return result.get("error", "No QR code found.")
    except Exception as e:
        logger.error(f"Error in scan_qr wrapper: {e}")
        return "Error scanning QR code"


def scan_qr_detailed(image_path: str) -> Dict[str, Any]:
    try:
        return qr_scanner.scan_qr(image_path)
    except Exception as e:
        logger.error(f"Error in scan_qr_detailed wrapper: {e}")
        return {"success": False, "error": str(e)}


def scan_qr_from_url(image_url: str) -> Dict[str, Any]:
    try:
        return qr_scanner.download_and_scan_qr(image_url)
    except Exception as e:
        logger.error(f"Error in scan_qr_from_url wrapper: {e}")
        return {"success": False, "error": str(e)}


def get_qr_info(qr_data: str) -> Dict[str, Any]:
    try:
        return {
            "data": qr_data,
            "type": qr_scanner._detect_qr_type(qr_data),
            "length": len(qr_data)
        }
    except Exception as e:
        logger.error(f"Error in get_qr_info wrapper: {e}")
        return {"error": str(e)}
