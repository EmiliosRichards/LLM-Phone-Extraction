from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import logging
from pathlib import Path
from typing import TypedDict, Union

class OCRResult(TypedDict):
    text: str
    char_count: int
    word_count: int
    path: str

def ocr_image(img_path: Path | str, enhancement: bool = True, fast_processing: bool = False) -> OCRResult:
    """Perform OCR on an image file.
    
    Args:
        img_path (Path | str): Path to the image file
        enhancement (bool, optional): Whether to apply contrast enhancement and sharpening. Defaults to True.
        fast_processing (bool, optional): If True, skips resizing for larger images (>1000x1000). Defaults to False.
        
    Returns:
        OCRResult: Dictionary containing:
            - text (str): Extracted text from the image
            - char_count (int): Number of characters extracted
            - word_count (int): Number of words extracted
            - path (str): Input image path as string
            
    Raises:
        ValueError: If the image is empty or corrupted
    """
    try:
        logging.debug(f"Starting OCR processing for {img_path}")
        img = Image.open(str(img_path)).convert('RGB')
        
        # Log image format and mode
        logging.debug(f"Image format: {img.format}, mode: {img.mode}")

        # Check if image is empty or corrupted
        if img.getbbox() is None:
            error_msg = f"Image appears to be empty or corrupted: {img_path}"
            logging.error(error_msg)
            raise ValueError(error_msg)

        # Convert to grayscale
        gray = img.convert('L')
        logging.debug(f"Image converted to grayscale: {gray.size}")

        # Resize (2x upscale if image is small)
        if not fast_processing or (gray.width < 1000 and gray.height < 1000):
            if gray.width < 300 or gray.height < 300:
                old_size = gray.size
                gray = gray.resize((gray.width * 2, gray.height * 2), Image.LANCZOS)
                logging.debug(f"Image upscaled from {old_size} to {gray.size}")
        else:
            logging.debug(f"Skipping resize for large image ({gray.width}x{gray.height}) due to fast_processing=True")

        # Improve contrast and sharpness if enhancement is enabled
        if enhancement:
            gray = ImageEnhance.Contrast(gray).enhance(2.0)  # Increase contrast
            gray = gray.filter(ImageFilter.SHARPEN)         # Apply sharpen filter
            logging.debug("Applied contrast enhancement and sharpening")
        else:
            logging.debug("Skipping image enhancement")

        # Run OCR
        text = pytesseract.image_to_string(gray)
        text_length = len(text)
        word_count = len(text.split())
        logging.info(f"OCR completed: extracted {text_length} characters, {word_count} words")
        
        if text_length == 0:
            logging.warning(f"No text extracted from {img_path}")
        
        return {
            "text": text,
            "char_count": text_length,
            "word_count": word_count,
            "path": str(img_path)
        }
    except FileNotFoundError as e:
        logging.error(f"Image file not found: {img_path} - {str(e)}")
        return {
            "text": "",
            "char_count": 0,
            "word_count": 0,
            "path": str(img_path)
        }
    except pytesseract.TesseractError as e:
        logging.error(f"Tesseract OCR error for {img_path}: {str(e)}")
        return {
            "text": "",
            "char_count": 0,
            "word_count": 0,
            "path": str(img_path)
        }
    except ValueError as e:
        logging.error(str(e))
        return {
            "text": "",
            "char_count": 0,
            "word_count": 0,
            "path": str(img_path)
        }
    except Exception as e:
        logging.error(f"Unexpected error during OCR for {img_path}: {str(e)}")
        return {
            "text": "",
            "char_count": 0,
            "word_count": 0,
            "path": str(img_path)
        }
